// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  Bogo GPU Worker — Universal / OpenCL — lease/range API v5 — v1.0                ║
// ║                                                                            ║
// ║  Same worker as the NVIDIA/CUDA TURBO build, retargeted to OpenCL so it    ║
// ║  runs on AMD, NVIDIA and Intel GPUs (any OpenCL) on Windows and Linux with     ║
// ║  nothing to install beyond the GPU driver. The compute engine and the      ║
// ║  shuffle math are byte-identical to the official engine (validated by      ║
// ║  bench_universal against the CPU reference) -> 0 rejected.                        ║
// ║                                                                            ║
// ║  Host side (websocket lease protocol, dashboard, fail-states, credentials  ║
// ║  from environment variables) is the same as the CUDA worker; only the GPU  ║
// ║  layer is OpenCL (cl_engine.h). See README.md.                             ║
// ║                                                                            ║
// ║    set BOGO_UUID=...   set BOGO_NICKNAME=...   set BOGO_CODE=...            ║
// ╚══════════════════════════════════════════════════════════════════════════╝
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include "cl_engine.h"

using json = nlohmann::json;

// ─── ACCOUNT (from environment, never hardcoded) ─────────────────────────────
static std::string env_or_empty(const char* name) {
#ifdef _WIN32
    wchar_t wname[64]{};
    for (int i = 0; i < 63 && name[i]; ++i) wname[i] = static_cast<wchar_t>(name[i]);
    wchar_t wval[512];
    DWORD n = GetEnvironmentVariableW(wname, wval, 512);
    if (n == 0 || n >= 512) return std::string();
    int len = WideCharToMultiByte(CP_UTF8, 0, wval, static_cast<int>(n), nullptr, 0, nullptr, nullptr);
    if (len <= 0) return std::string();
    std::string out(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wval, static_cast<int>(n), &out[0], len, nullptr, nullptr);
    return out;
#else
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : std::string();
#endif
}
static const std::string UUID     = env_or_empty("BOGO_UUID");
static const std::string NICKNAME = env_or_empty("BOGO_NICKNAME");
static const std::string CODE     = env_or_empty("BOGO_CODE");
static const std::string WS_URL   = "wss://bogo.swapjs.dev/ws";

// ─── CONFIG ──────────────────────────────────────────────────────────────────
constexpr int NUM_CONNECTIONS = 1;
constexpr int NUM_WORKERS     = 1;
constexpr int NUM_SENDERS     = 1;
constexpr int REPORT_MS = 1000;
constexpr uint64_t STOP_AT_LIFETIME = 1000ULL * 100000000000000000ULL;  // ~never
constexpr const char* WORKER_VERSION = "Universal v1.0";

// ─── STATE ───────────────────────────────────────────────────────────────────
std::atomic<bool> running{true};
std::atomic<bool> ws_open{false};
std::atomic<uint64_t> global_shuffles{0};
std::atomic<uint64_t> global_credit{0};
std::atomic<uint64_t> global_reports{0};
std::atomic<uint64_t> global_leases{0};
std::atomic<uint64_t> global_rejected{0};
std::atomic<uint64_t> global_reconnects{0};
std::atomic<int> open_count{0};
std::atomic<bool> got_welcome{false};
std::atomic<uint64_t> current_lease_done{0};
std::atomic<uint64_t> current_lease_count{0};
std::atomic<double> gpu_rate{0.0};
std::atomic<bool> tester_mode{false};

std::mutex statusMutex;
std::string statusLine = "starting";
std::string lastServerMessage = "";
std::string deviceLine = "(detecting GPU...)";
std::string tuneLine = "(pending)";
std::chrono::steady_clock::time_point programStart;

static double since_start_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - programStart).count();
}
static void debug_log(const std::string& line) {
    if (!tester_mode.load(std::memory_order_relaxed)) return;
    std::cerr << "[" << std::fixed << std::setprecision(3) << since_start_s() << "s] " << line << std::endl;
}
static std::string comma_u64(uint64_t n) {
    std::string s = std::to_string(n);
    int p = static_cast<int>(s.length()) - 3;
    while (p > 0) { s.insert(static_cast<size_t>(p), ","); p -= 3; }
    return s;
}
static std::string rate_string(double r) {
    std::ostringstream o;
    if (r >= 1e9) o << std::fixed << std::setprecision(2) << r / 1e9 << " B/s";
    else if (r >= 1e6) o << std::fixed << std::setprecision(2) << r / 1e6 << " M/s";
    else o << std::fixed << std::setprecision(0) << r << " /s";
    return o.str();
}
// Wall-clock since program start, formatted HH:MM:SS (with a leading "Nd " for
// runs past a day) for the dashboard's TimeElapsed line.
static std::string elapsed_string(double s) {
    uint64_t total = (uint64_t)(s < 0.0 ? 0.0 : s);
    uint64_t days = total / 86400, hours = (total % 86400) / 3600;
    uint64_t mins = (total % 3600) / 60, secs = total % 60;
    std::ostringstream o;
    if (days > 0) o << days << "d ";
    o << std::setfill('0') << std::setw(2) << hours << ":"
      << std::setw(2) << mins << ":" << std::setw(2) << secs;
    return o.str();
}
static void set_status(const std::string& s) { std::lock_guard<std::mutex> l(statusMutex); statusLine = s; }
static void set_server_message(const std::string& s) { std::lock_guard<std::mutex> l(statusMutex); lastServerMessage = s; }
static void set_device_line(const std::string& s) { std::lock_guard<std::mutex> l(statusMutex); deviceLine = s; }
static void set_tune_line(const std::string& s) { std::lock_guard<std::mutex> l(statusMutex); tuneLine = s; }
static std::string json_redacted(json j) { if (j.contains("code")) j["code"] = "***"; return j.dump(); }

// ─── FAIL STATES ─────────────────────────────────────────────────────────────
std::atomic<bool> had_fatal{false};
std::string fatalMessage;
static void fail(const std::string& msg);

static bool utf8_valid(const std::string& s) {
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        int ext = (c < 0x80) ? 0 : (c >= 0xC2 && c <= 0xDF) ? 1
                : (c >= 0xE0 && c <= 0xEF) ? 2 : (c >= 0xF0 && c <= 0xF4) ? 3 : -1;
        if (ext < 0 || i + static_cast<size_t>(ext) >= n) return false;
        for (int k = 1; k <= ext; ++k)
            if ((static_cast<unsigned char>(s[i + static_cast<size_t>(k)]) & 0xC0) != 0x80) return false;
        i += static_cast<size_t>(ext) + 1;
    }
    return true;
}
static size_t utf8_length(const std::string& s) {
    size_t n = 0;
    for (char c : s) if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) n++;
    return n;
}
static void wait_for_enter() {
    if (tester_mode.load(std::memory_order_relaxed)) return;
    std::cout << "\nPress Enter to exit..." << std::flush;
    std::cin.clear();
    std::string line;
    std::getline(std::cin, line);
}

// ─── QUEUES ──────────────────────────────────────────────────────────────────
struct QueuedLease { std::string seed; uint64_t count = 0; int conn_idx = 0; };
std::queue<QueuedLease> jobQueue;
std::mutex jobMutex;
std::condition_variable jobCV;

struct QueuedResult { int conn_idx = 0; std::string payload; };
std::queue<QueuedResult> resultQueue;
std::mutex resultMutex;
std::condition_variable resultCV;

struct alignas(64) PaddedMutex { std::mutex m; };
ix::WebSocket* connections[NUM_CONNECTIONS]{};
PaddedMutex sendMutex[NUM_CONNECTIONS];

static void queue_send(int conn_idx, const json& payload) {
    debug_log("SEND conn=" + std::to_string(conn_idx) + " " + json_redacted(payload));
    { std::lock_guard<std::mutex> l(resultMutex); resultQueue.push({conn_idx, payload.dump()}); }
    resultCV.notify_one();
}

static void fail(const std::string& msg) {
    { std::lock_guard<std::mutex> l(statusMutex); if (fatalMessage.empty()) fatalMessage = msg; }
    had_fatal.store(true, std::memory_order_relaxed);
    set_status("ERROR: " + msg);
    running.store(false, std::memory_order_relaxed);
    jobCV.notify_all();
    resultCV.notify_all();
}

// ─── COMPUTE WORKER (OpenCL) ─────────────────────────────────────────────────
void worker_thread(int) {
    ClEngine eng;
    ClConfig cfg;
    if (!eng.init(cfg)) {
        fail(std::string("no usable OpenCL GPU - ") + eng.lastError +
             " (a GPU with an OpenCL driver is required)");
        return;
    }
    set_device_line(eng.deviceName + "  (" + std::to_string(eng.computeUnits) + " CU, " +
                    std::to_string(eng.threads) + " work-items)");
    debug_log("OpenCL device: " + eng.deviceName + " / " + eng.vendor);

    // ── Per-card auto-tuning ─────────────────────────────────────────────────
    // Probe a few launch shapes (WG / total work-items / chunk) on this exact GPU
    // and keep the fastest, so the worker runs near the card's optimum without any
    // manual env tuning. Safe: the launch shape never changes the bit-exact result
    // (see ClEngine::autotune) — it only moves throughput. Disable with
    // BOGO_AUTOTUNE=0; tune length is BOGO_AUTOTUNE_SECS (default 12), repeats per
    // config BOGO_AUTOTUNE_REPS (default 3). A longer run measures more carefully.
    if (ClEngine::envl("BOGO_AUTOTUNE", 1) != 0) {
        set_status("auto-tuning for this GPU (measuring best values)...");
        double secs = (double)ClEngine::envl("BOGO_AUTOTUNE_SECS", 12);
        if (secs < 1.0) secs = 12.0;
        std::string sum = eng.autotune(secs, [](const std::string& s){ set_status(s); });
        set_tune_line(sum);
        set_device_line(eng.deviceName + "  (" + std::to_string(eng.computeUnits) + " CU, " +
                        std::to_string(eng.threads) + " work-items)");
        set_status("waiting for lease");
        debug_log("autotune -> " + sum);
    } else {
        set_tune_line("off (BOGO_AUTOTUNE=0; using env/defaults)");
    }

    const uint64_t CHUNK = eng.cfg.CHUNK;

    // Trailing-window throughput (persists across leases) for a STABLE displayed
    // rate. Each launch is short and its wall time carries variable host/driver
    // overhead plus the periodic low-floor baseline chunk, so a per-chunk
    // instantaneous rate swings 2x+ ("20..40 B/s and back"); averaging the real
    // shuffles-per-second over a ~3s window is the true sustained rate, and steady.
    std::deque<std::pair<std::chrono::steady_clock::time_point, uint64_t>> rateWin;
    uint64_t rateCum = 0;
    rateWin.push_back({ std::chrono::steady_clock::now(), 0 });

    while (running.load(std::memory_order_relaxed)) {
        QueuedLease lease;
        {
            std::unique_lock<std::mutex> lock(jobMutex);
            jobCV.wait(lock, [] { return !jobQueue.empty() || !running.load(); });
            if (!running.load() && jobQueue.empty()) break;
            lease = std::move(jobQueue.front());
            jobQueue.pop();
        }
        try {
            global_leases.fetch_add(1, std::memory_order_relaxed);
            current_lease_done.store(0, std::memory_order_relaxed);
            current_lease_count.store(lease.count, std::memory_order_relaxed);
            set_status("computing lease");

            const uint64_t base_seed = std::stoull(lease.seed);
            uint64_t totalDone = 0, lastReported = 0, winIndex = 0;
            int winBest = -1;
            std::array<uint8_t, 25> winArr{};
            auto lastReportTime = std::chrono::steady_clock::now();

            while (running.load(std::memory_order_relaxed) && totalDone < lease.count) {
                if (!ws_open.load(std::memory_order_relaxed)) {
                    debug_log("lease aborted: connection lost; waiting for a fresh lease");
                    break;
                }
                const uint64_t lo = totalDone;
                const uint64_t hi = std::min<uint64_t>(lo + CHUNK, lease.count);

                ClTriple rr = eng.compute_range(base_seed, lo, hi,
                        winBest >= 0 ? winBest : cfg.BEST_FLOOR, winBest < 0);
                if (!eng.lastError.empty()) { fail("OpenCL compute error: " + eng.lastError); break; }

                const uint64_t scanned = hi - lo;
                totalDone = hi;
                current_lease_done.store(totalDone, std::memory_order_relaxed);
                global_shuffles.fetch_add(scanned, std::memory_order_relaxed);

                // Stable rate: shuffles/second over a trailing ~3s window (see above).
                rateCum += scanned;
                const auto rnow = std::chrono::steady_clock::now();
                if (!rateWin.empty() &&
                    std::chrono::duration<double>(rnow - rateWin.back().first).count() > 5.0)
                    rateWin.clear();                              // long gap (reconnect/idle): restart the window
                rateWin.push_back({ rnow, rateCum });
                while (rateWin.size() > 1 &&
                       std::chrono::duration<double>(rnow - rateWin.front().first).count() > 3.0)
                    rateWin.pop_front();
                if (rateWin.size() >= 2) {
                    double dt = std::chrono::duration<double>(rateWin.back().first - rateWin.front().first).count();
                    if (dt > 0.0)
                        gpu_rate.store((double)(rateWin.back().second - rateWin.front().second) / dt,
                                       std::memory_order_relaxed);
                }

                if (rr.best > winBest) { winBest = rr.best; winArr = rr.arr; winIndex = rr.idx; }

                const auto now = std::chrono::steady_clock::now();
                const bool reportDue = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastReportTime).count() >= REPORT_MS;
                const bool leaseDone = (totalDone >= lease.count);
                if ((reportDue || leaseDone) && totalDone > lastReported && winBest >= 0) {
                    json arr = json::array();
                    for (uint8_t v : winArr) arr.push_back(static_cast<int>(v));
                    json payload = {
                        {"type", "result"}, {"seed", lease.seed}, {"total_done", totalDone},
                        {"best_correct", winBest}, {"best_arr", arr}, {"best_index", winIndex}
                    };
                    queue_send(lease.conn_idx, payload);
                    global_reports.fetch_add(1, std::memory_order_relaxed);
                    lastReported = totalDone; lastReportTime = now;
                    winBest = -1; winArr = {}; winIndex = 0;
                }
            }
            set_status("waiting for next lease");
        } catch (const std::exception& e) {
            fail(std::string("worker error: ") + e.what());
        }
    }
    eng.destroy();
}

// ─── SENDER ──────────────────────────────────────────────────────────────────
void sender_thread() {
    while (running.load(std::memory_order_relaxed)) {
        QueuedResult res;
        {
            std::unique_lock<std::mutex> lock(resultMutex);
            resultCV.wait(lock, [] { return !resultQueue.empty() || !running.load(); });
            if (!running.load() && resultQueue.empty()) return;
            res = std::move(resultQueue.front());
            resultQueue.pop();
        }
        if (res.conn_idx < 0 || res.conn_idx >= NUM_CONNECTIONS || !connections[res.conn_idx]) continue;
        std::lock_guard<std::mutex> slock(sendMutex[res.conn_idx].m);
        connections[res.conn_idx]->send(res.payload);
    }
}

// ─── WEBSOCKET ───────────────────────────────────────────────────────────────
ix::WebSocket* make_connection(int idx) {
    auto* ws = new ix::WebSocket();
    ws->setUrl(WS_URL);
    ws->enableAutomaticReconnection();
    ws->setMinWaitBetweenReconnectionRetries(1000);
    ws->setMaxWaitBetweenReconnectionRetries(15000);
    ws->setHandshakeTimeout(10);
    ws->setPingInterval(20);

    ix::WebSocketHttpHeaders headers;
    headers["Origin"] = "https://bogo.swapjs.dev";
    headers["Referer"] = "https://bogo.swapjs.dev/contribute";
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BogoOpenCLClient/1.0";
    ws->setExtraHeaders(headers);

    ws->setOnMessageCallback([idx](const ix::WebSocketMessagePtr& msg) {
      try {
        if (msg->type == ix::WebSocketMessageType::Open) {
            ws_open.store(true, std::memory_order_relaxed);
            if (open_count.fetch_add(1) > 0) {
                global_reconnects.fetch_add(1, std::memory_order_relaxed);
                set_status("reconnected; re-sending hello");
            } else {
                set_status("websocket open; sending hello");
            }
            json hello = { {"type", "hello"}, {"v", 5}, {"uuid", UUID}, {"nickname", NICKNAME}, {"code", CODE} };
            debug_log("SEND HELLO " + json_redacted(hello));
            std::lock_guard<std::mutex> slock(sendMutex[idx].m);
            connections[idx]->send(hello.dump());
        } else if (msg->type == ix::WebSocketMessageType::Message) {
            debug_log("RECV " + msg->str.substr(0, 200));
            try {
                json data = json::parse(msg->str);
                const std::string type = data.value("type", "");
                if (type == "welcome") {
                    got_welcome.store(true, std::memory_order_relaxed);
                    uint64_t lifetime = data.value("lifetime_shuffles", (uint64_t)0);
                    set_server_message("welcome; lifetime=" + comma_u64(lifetime));
                    set_status("waiting for lease");
                } else if (type == "job") {
                    const std::string seed = data.at("seed").get<std::string>();
                    const uint64_t count = data.at("count").get<uint64_t>();
                    { std::lock_guard<std::mutex> lock(jobMutex); jobQueue.push({seed, count, idx}); }
                    jobCV.notify_one();
                    set_server_message("lease; count=" + comma_u64(count));
                } else if (type == "credited") {
                    uint64_t credit = data.value("credit", (uint64_t)0);
                    global_credit.fetch_add(credit, std::memory_order_relaxed);
                    int bb = data.value("batch_best", -1);
                    set_server_message("credited +" + comma_u64(credit) + "; batch_best=" + std::to_string(bb));
                    uint64_t lifetime = data.value("lifetime_shuffles", (uint64_t)0);
                    if (STOP_AT_LIFETIME && lifetime >= STOP_AT_LIFETIME) {
                        set_server_message("reached lifetime target - stopping");
                        if (connections[idx]) connections[idx]->disableAutomaticReconnection();
                        running.store(false, std::memory_order_relaxed);
                        jobCV.notify_all(); resultCV.notify_all();
                    }
                } else if (type == "rejected") {
                    const std::string reason = data.value("reason", msg->str.substr(0, 200));
                    if (!got_welcome.load(std::memory_order_relaxed)) {
                        if (connections[idx]) connections[idx]->disableAutomaticReconnection();
                        fail("server rejected the login: " + reason);
                    } else {
                        global_rejected.fetch_add(1, std::memory_order_relaxed);
                        set_server_message("rejected: " + reason);
                    }
                } else if (type == "client_outdated") {
                    set_server_message("client_outdated: " + data.value("message", ""));
                    set_status("client outdated; stopping");
                    if (connections[idx]) connections[idx]->disableAutomaticReconnection();
                    running.store(false); jobCV.notify_all(); resultCV.notify_all();
                } else if (type == "banned") {
                    set_server_message("banned: " + data.value("reason", "unknown"));
                    if (connections[idx]) connections[idx]->disableAutomaticReconnection();
                    running.store(false); jobCV.notify_all(); resultCV.notify_all();
                } else if (type == "contributions_closed") {
                    set_server_message("contributions closed");
                    if (connections[idx]) connections[idx]->disableAutomaticReconnection();
                    running.store(false); jobCV.notify_all(); resultCV.notify_all();
                }
            } catch (const std::exception& e) {
                set_server_message(std::string("parse error: ") + e.what());
            }
        } else if (msg->type == ix::WebSocketMessageType::Close) {
            ws_open.store(false, std::memory_order_relaxed);
            if (running.load(std::memory_order_relaxed)) set_status("disconnected; reconnecting");
            set_server_message("closed code=" + std::to_string(msg->closeInfo.code) + " " +
                               msg->closeInfo.reason + " (auto-reconnecting)");
            jobCV.notify_all(); resultCV.notify_all();
        } else if (msg->type == ix::WebSocketMessageType::Error) {
            ws_open.store(false, std::memory_order_relaxed);
            if (running.load(std::memory_order_relaxed)) set_status("connection failed; retrying");
            set_server_message("error: " + msg->errorInfo.reason + " status=" +
                               std::to_string(msg->errorInfo.http_status) + " (retrying)");
            jobCV.notify_all(); resultCV.notify_all();
        }
      } catch (const std::exception& e) {
        fail(std::string("websocket handler error: ") + e.what());
      }
    });
    return ws;
}

// ─── DASHBOARD ───────────────────────────────────────────────────────────────
void dashboard_thread() {
    std::cout << "\x1b[2J";
    while (running.load(std::memory_order_relaxed)) {
        uint64_t done = current_lease_done.load(), count = current_lease_count.load();
        double pct = count > 0 ? 100.0 * (double)done / (double)count : 0.0;
        std::string status, server, device, tune;
        { std::lock_guard<std::mutex> l(statusMutex); status = statusLine; server = lastServerMessage; device = deviceLine; tune = tuneLine; }
        if (server.size() > 80) server.resize(80);
        std::cout << "\x1b[H";
        std::cout << "=== BOGOSORT UNIVERSAL GPU WORKER (OpenCL) " << WORKER_VERSION << " (lease API v5, h-mask) ===\n";
        std::cout << "--- Made by: .Maf (discord) ---\n";
        std::cout << "Name:        " << NICKNAME << "\n";
        std::cout << "GPU:         " << device << "          \n";
        std::cout << "Tuned:       " << tune << "          \n";
        std::cout << "WebSocket:   " << (ws_open.load() ? "open" : "closed") << "\n";
        std::cout << "TimeElapsed: " << elapsed_string(since_start_s()) << "          \n";
        std::cout << "Kernel rate: " << rate_string(gpu_rate.load()) << "          \n";
        std::cout << "Session:     " << comma_u64(global_shuffles.load()) << "          \n";
        std::cout << "Credited:    " << comma_u64(global_credit.load()) << "          \n";
        std::cout << "Reports:     " << comma_u64(global_reports.load()) << "   Leases: " << comma_u64(global_leases.load())
                  << "   Rejected: " << comma_u64(global_rejected.load())
                  << "   Reconnects: " << comma_u64(global_reconnects.load()) << "      \n";
        std::cout << "Lease:       " << comma_u64(done) << " / " << comma_u64(count)
                  << " (" << std::fixed << std::setprecision(1) << pct << "%)        \n";
        std::cout << "Status:      " << status << "          \n";
        std::cout << "Server:      " << server << "          \n";
        std::cout << "===========================================\n" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}

// ─── MAIN ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    programStart = std::chrono::steady_clock::now();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tester") tester_mode.store(true);
        else if (a == "--help" || a == "-h") { std::cout << "Usage: " << argv[0] << " [--tester]\n"; return 0; }
    }
    if (UUID.empty() || NICKNAME.empty() || CODE.empty()) {
        std::cerr << "[ERROR] Missing credentials.\n"
                     "        Set the BOGO_UUID, BOGO_NICKNAME and BOGO_CODE environment\n"
                     "        variables, or run start_universal.bat / start_universal.sh which ask for them.\n";
        wait_for_enter();
        return 1;
    }
    if (!utf8_valid(NICKNAME) || !utf8_valid(UUID) || !utf8_valid(CODE)) {
        std::cerr << "[ERROR] Credentials contain bytes that are not valid UTF-8 text.\n"
                     "        Use plain ASCII characters.\n";
        wait_for_enter();
        return 1;
    }
    if (utf8_length(NICKNAME) > 8) {
        std::cerr << "[ERROR] Nickname \"" << NICKNAME << "\" is " << utf8_length(NICKNAME)
                  << " characters long.\n"
                     "        The server requires 8 characters or fewer - pick a shorter one.\n";
        wait_for_enter();
        return 1;
    }

    ix::initNetSystem();
    std::cout << "Bogo OpenCL worker " << WORKER_VERSION << " (lease/range API v5, OpenCL h-mask kernel)\n"
              << "Target: " << WS_URL << "\nNickname: " << NICKNAME << "\n"
              << "Starting OpenCL (set BOGO_PLATFORM / BOGO_DEVICE to pick a GPU; "
                 "BOGO_THREADS / BOGO_WG / BOGO_CHUNK_LOG to tune)\n";

    std::vector<std::thread> workers, senders;
    for (int i = 0; i < NUM_WORKERS; ++i) workers.emplace_back(worker_thread, i);
    for (int i = 0; i < NUM_SENDERS; ++i) senders.emplace_back(sender_thread);
    std::thread dash;
    if (!tester_mode.load()) dash = std::thread(dashboard_thread);

    for (int i = 0; i < NUM_CONNECTIONS; ++i) {
        connections[i] = make_connection(i);
        connections[i]->start();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    while (running.load(std::memory_order_relaxed)) std::this_thread::sleep_for(std::chrono::milliseconds(250));

    for (int i = 0; i < NUM_CONNECTIONS; ++i) {
        if (connections[i]) {
            try { json stop = {{"type", "stop"}};
                  std::lock_guard<std::mutex> slock(sendMutex[i].m);
                  connections[i]->send(stop.dump()); } catch (...) {}
        }
    }
    jobCV.notify_all(); resultCV.notify_all();
    for (auto& t : workers) if (t.joinable()) t.join();
    for (auto& t : senders) if (t.joinable()) t.join();
    for (int i = 0; i < NUM_CONNECTIONS; ++i) { if (connections[i]) { connections[i]->stop(); delete connections[i]; } }
    if (dash.joinable()) dash.join();
    ix::uninitNetSystem();

    std::string status, server, fatalMsg;
    {
        std::lock_guard<std::mutex> l(statusMutex);
        status = statusLine; server = lastServerMessage; fatalMsg = fatalMessage;
    }
    std::cout << "\n=== WORKER STOPPED ===\n";
    if (!fatalMsg.empty()) std::cout << "Error:  " << fatalMsg << "\n";
    if (!server.empty())   std::cout << "Server: " << server << "\n";
    std::cout << "Status: " << status << "\n";
    wait_for_enter();
    return had_fatal.load() ? 1 : 0;
}
