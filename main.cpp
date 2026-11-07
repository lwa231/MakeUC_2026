// Brace Tracker viewer: live 3D knee (thigh + shin on a hinge).
//
// Live data: this program launches a helper (default: fake_pi.py) and reads its
// stdout, one line per sample at ~100 Hz, comma-separated, in g:
//     t_ms,thigh_x,thigh_y,thigh_z,shank_x,shank_y,shank_z
// All the math is done here in C++: low-pass filter, tilt of each segment, knee
// angle (shank tilt - thigh tilt), quality, and zero calibration.
// To use the real Pi, replace fake_pi.py with anything that prints the same lines
// (flush every line), or pass --pi-cmd "your command".
//
// Rendering: a tapered thigh and calf (lathe meshes sized from the circumferences),
// a knee ball and a foot. Every part is derived from the four measurements and one
// overall size scale, so changing any one number resizes everything that depends on it.
// A seat and floor give context (kShowSeatAndFloor). Soft lighting, 4x MSAA.
//
// Angle source, in priority order:
//   LIVE        from the helper process
//   AUTO DEMO   built-in sweep until live data arrives
//   MANUAL      UP/DOWN keys (after turning the demo off with SPACE)
// If live data stops for >0.5 s the last angle is held ("SIGNAL LOST").
//
// Body measurements (tape measure, in cm) can be set three ways:
//   1. Edit the defaults in LegState below.
//   2. Command line:  viewer --thigh-circ 56 --calf-circ 38 --thigh-len 46 --shin-len 43
//   3. Live with the keys below.
//
// Controls
//   Z             zero-calibrate: hold the leg straight for 2 s (live data only)
//   UP / DOWN     knee flexion +/-  (manual only; 0 deg = straight, flexion positive)
//   Q / A         thigh circumference +/-
//   W / S         calf circumference +/-
//   - / =         overall leg size (scales every part together)
//   LEFT / RIGHT  orbit camera
//   SPACE         toggle the auto demo sweep
//   R             reset angle and camera
//   ESC           quit
//
// Options: --pi-cmd "command"   (default: "python fake_pi.py", or python3 off Windows)
//          --scale s            (overall leg size, 1.0 = as measured)

#ifdef _MSC_VER
  #define _CRT_SECURE_NO_WARNINGS
#endif

#include <glad/glad.h>
#include <GLFW/include/GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
  #define popen _popen
  #define pclose _pclose
#endif

constexpr float kPi = 3.14159265358979f;

// ======================= live data from the helper process =======================
// ---- live data (begin) ----

// Axis map: which sensor axes (0=x, 1=y, 2=z) lie in the leg's plane of motion /
// point up at rest, and their signs. Fix these on the first test on the leg.
constexpr int kSagAxis = 0, kVertAxis = 2;
constexpr float kSagSign = 1.0f, kVertSign = 1.0f;
constexpr float kKneeSign = 1.0f;   // -1 if bending the knee makes the angle negative
constexpr float kCutoffHz = 5.0f;   // low-pass cutoff
constexpr double kZeroMs = 2000.0;  // hold the leg straight this long to zero

static std::atomic<bool> g_running{true};
static std::atomic<bool> g_zeroRequested{false};
static std::atomic<bool> g_zeroing{false};
static std::atomic<float> g_liveKnee{0.0f};
static std::atomic<float> g_liveQuality{0.0f};
static std::atomic<long long> g_lastLiveMs{-1};  // steady-clock ms of the last sample

static long long nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static float wrapDeg(float a) { return std::fmod(std::fmod(a + 180.0f, 360.0f) + 360.0f, 360.0f) - 180.0f; }

// Tilt of one segment from its gravity vector a[3] (g).
static float tiltDeg(const float* a) {
    return std::atan2(kSagSign * a[kSagAxis], kVertSign * a[kVertAxis]) * 180.0f / kPi;
}

// 0..1: how close |a| is to 1 g (fast motion adds acceleration and corrupts the tilt).
static float qualityOf(const float* a) {
    float mag = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    return std::max(0.0f, 1.0f - std::fabs(mag - 1.0f) / 0.3f);
}

static void liveReaderThread(std::string cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::fprintf(stderr, "Could not start: %s\n", cmd.c_str());
        return;
    }
    std::printf("Reading live data from: %s\n", cmd.c_str());

    float y[6] = {0};          // low-pass state: thigh xyz, shank xyz
    bool haveY = false;
    double lastT = -1.0;
    float offset = 0.0f;       // zero calibration
    bool zeroing = false;
    double zeroEnd = 0.0, zSin = 0.0, zCos = 0.0;

    char line[256];
    while (g_running && std::fgets(line, sizeof(line), pipe)) {
        double tMs;
        float v[6];
        if (std::sscanf(line, "%lf,%f,%f,%f,%f,%f,%f", &tMs, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 7)
            continue;  // ignore any line that is not a sample

        float dt = (lastT < 0.0) ? 0.01f : (float)((tMs - lastT) / 1000.0);
        lastT = tMs;
        if (dt <= 0.0f || dt > 0.5f) dt = 0.01f;
        const float rc = 1.0f / (2.0f * kPi * kCutoffHz);
        const float alpha = dt / (rc + dt);

        const float q = std::min(qualityOf(v), qualityOf(v + 3));  // unfiltered: motion shows up here
        if (!haveY) { std::copy(v, v + 6, y); haveY = true; }
        else for (int i = 0; i < 6; ++i) y[i] += alpha * (v[i] - y[i]);

        const float raw = wrapDeg(kKneeSign * (tiltDeg(y + 3) - tiltDeg(y)));  // shank - thigh

        if (g_zeroRequested.exchange(false)) {
            zeroing = true;
            zeroEnd = tMs + kZeroMs;
            zSin = zCos = 0.0;
            std::printf("ZEROING: hold the leg straight for 2 s...\n");
        }
        if (zeroing) {
            zSin += std::sin(raw * kPi / 180.0f);
            zCos += std::cos(raw * kPi / 180.0f);
            if (tMs >= zeroEnd) {
                offset = (float)(std::atan2(zSin, zCos) * 180.0 / kPi);
                zeroing = false;
                std::printf("Zero set: offset %+.2f deg\n", offset);
            }
        }
        g_zeroing = zeroing;

        g_liveKnee = wrapDeg(raw - offset);
        g_liveQuality = q;
        g_lastLiveMs = nowMs();
    }
    pclose(pipe);
}

// ---- live data (end) ----
// ================================================================================

// ---------------------------------------------------------------- state ----

constexpr float kUnitsPerCm = 0.035f;  // world units per centimeter (keeps the leg ~3 units long)
constexpr bool kShowSeatAndFloor = true;  // false = draw only the leg
constexpr float kMaxKneeDeg = 120.0f;     // display limit; the seat is sized so the calf clears it up to here

struct LegState {
    float kneeDeg = 0.0f;         // 0 = full extension, flexion positive

    // ---- body measurements in cm: PUT THE PATIENT'S VALUES HERE ----
    float thighCircCm = 55.0f;    // circumference around the widest part of the thigh
    float calfCircCm = 37.0f;     // circumference around the widest part of the calf
    float thighLenCm = 45.0f;     // hip -> knee
    float shinLenCm = 42.0f;      // knee -> ankle

    float scale = 1.0f;           // overall leg size: scales every part together (1 = as measured)
};

static LegState g_leg;
static bool g_autoSweep = true;
static float g_camYawDeg = 0.0f;
static int g_fbWidth = 1000, g_fbHeight = 700;
#ifdef _WIN32
static std::string g_piCmd = "python fake_pi.py";
#else
static std::string g_piCmd = "python3 fake_pi.py";
#endif

// Circumference -> diameter, assuming a roughly circular cross-section.
static float diameterCm(float circCm) { return circCm / kPi; }

static void clampLeg() {
    g_leg.kneeDeg = std::clamp(g_leg.kneeDeg, 0.0f, kMaxKneeDeg);
    g_leg.thighCircCm = std::clamp(g_leg.thighCircCm, 20.0f, 100.0f);
    g_leg.calfCircCm = std::clamp(g_leg.calfCircCm, 15.0f, 80.0f);
    g_leg.thighLenCm = std::clamp(g_leg.thighLenCm, 20.0f, 80.0f);
    g_leg.shinLenCm = std::clamp(g_leg.shinLenCm, 20.0f, 80.0f);
    g_leg.scale = std::clamp(g_leg.scale, 0.6f, 1.6f);
}

static void parseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        float* target = nullptr;
        if (!std::strcmp(argv[i], "--thigh-circ")) target = &g_leg.thighCircCm;
        else if (!std::strcmp(argv[i], "--calf-circ")) target = &g_leg.calfCircCm;
        else if (!std::strcmp(argv[i], "--thigh-len")) target = &g_leg.thighLenCm;
        else if (!std::strcmp(argv[i], "--shin-len")) target = &g_leg.shinLenCm;
        else if (!std::strcmp(argv[i], "--scale")) target = &g_leg.scale;

        if (target && i + 1 < argc) {
            *target = (float)std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--pi-cmd") && i + 1 < argc) {
            g_piCmd = argv[++i];
        } else {
            std::fprintf(stderr,
                         "Unknown or incomplete option '%s'\n"
                         "Usage: viewer [--thigh-circ cm] [--calf-circ cm] [--thigh-len cm] [--shin-len cm] [--scale s]\n"
                         "              [--pi-cmd \"command that prints samples\"]\n",
                         argv[i]);
        }
    }
    clampLeg();
}

// -------------------------------------------------------------- shaders ----

static const char* kVertSrc = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uModel;
uniform mat4 uViewProj;
out vec3 vNormal;
out vec3 vWorldPos;
void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    gl_Position = uViewProj * world;
}
)";

// Soft key light (wrapped diffuse so skin does not look harsh), a cool fill light,
// sky/ground ambient, Blinn-Phong highlight and a faint rim. Colors are treated as
// sRGB: linearized on the way in, gamma-corrected on the way out.
static const char* kFragSrc = R"(#version 330 core
in vec3 vNormal;
in vec3 vWorldPos;
uniform vec3 uColor;
uniform vec3 uViewPos;
uniform float uSpec;
uniform float uShininess;
out vec4 FragColor;
void main() {
    vec3 albedo = pow(uColor, vec3(2.2));
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uViewPos - vWorldPos);
    vec3 L1 = normalize(vec3(0.5, 0.9, 0.7));
    vec3 L2 = normalize(vec3(-0.7, 0.3, 0.5));
    float d1 = max((dot(N, L1) + 0.25) / 1.25, 0.0);
    float d2 = max(dot(N, L2), 0.0);
    vec3 hemi = mix(vec3(0.03, 0.03, 0.04), vec3(0.12, 0.13, 0.17), 0.5 + 0.5 * N.y);
    vec3 H = normalize(L1 + V);
    float spec = uSpec * pow(max(dot(N, H), 0.0), uShininess);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 col = albedo * (hemi + 0.95 * d1 + 0.18 * d2 * vec3(0.7, 0.8, 1.0))
             + vec3(spec) + 0.10 * rim * vec3(0.6, 0.7, 1.0);
    FragColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
)";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Shader compile error:\n%s\n", log);
    }
    return s;
}

static GLuint makeProgram() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, kVertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragSrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Program link error:\n%s\n", log);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

// ----------------------------------------------------------------- mesh ----
// ---- mesh (begin) ----
// Every mesh is a triangle list, each vertex (x y z nx ny nz), axis along X from
// -0.5 to +0.5 and diameter 1, so a model matrix scale(length, diameter, diameter)
// sizes it.

static float lerpf(float a, float b, float t) { return a + (b - a) * t; }

static float smooth01(float a, float b, float x) {
    float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Rounds the start (u near 0) of a profile over width w with a quarter-circle.
static float capStart(float u, float w) {
    if (u >= w) return 1.0f;
    float t = 1.0f - u / w;
    return std::sqrt(std::max(0.0f, 1.0f - t * t));
}
static float capEnd(float u, float w) { return capStart(1.0f - u, w); }

// Radius multipliers along each segment (1.0 = the measured circumference), u: 0 -> 1.
// Thigh: hip (u=0) -> knee (u=1), widest in the upper third, narrowing toward the knee.
static float thighProfile(float u) {
    return lerpf(1.0f, 0.72f, smooth01(0.30f, 1.0f, u)) * capStart(u, 0.12f);
}
// Shin: knee (u=0) -> ankle (u=1), calf bulge near the top, narrow ankle.
static float shinProfile(float u) {
    float base = (u < 0.28f) ? lerpf(0.88f, 1.0f, smooth01(0.0f, 0.28f, u))
                             : lerpf(1.0f, 0.62f, smooth01(0.28f, 1.0f, u));
    return base * capEnd(u, 0.07f);
}
// Sphere / ellipsoid.
static float ballProfile(float u) {
    float t = 2.0f * u - 1.0f;
    return std::sqrt(std::max(0.0f, 1.0f - t * t));
}

// Surface of revolution around X with radius 0.5 * profile(u). Normals come from the
// profile's slope, so the shading is smooth.
static std::vector<float> makeLathe(float (*profile)(float), int stations = 48, int sides = 40) {
    struct V { float x, y, z, nx, ny, nz; };
    std::vector<V> ring((size_t)(stations + 1) * (sides + 1));
    for (int i = 0; i <= stations; ++i) {
        float u = 0.5f - 0.5f * std::cos(kPi * (float)i / stations);  // denser near the ends
        float m = profile(u);
        const float e = 1e-3f;
        float ua = std::max(u - e, 0.0f), ub = std::min(u + e, 1.0f);
        float dm = (profile(ub) - profile(ua)) / (ub - ua);
        float nx = -0.5f * dm, nr = 1.0f;
        float len = std::sqrt(nx * nx + nr * nr);
        nx /= len;
        nr /= len;
        for (int j = 0; j <= sides; ++j) {
            float a = 2.0f * kPi * (float)j / sides;
            ring[(size_t)i * (sides + 1) + j] = {u - 0.5f, 0.5f * m * std::cos(a), 0.5f * m * std::sin(a),
                                                 nx, nr * std::cos(a), nr * std::sin(a)};
        }
    }
    std::vector<float> out;
    out.reserve((size_t)stations * sides * 6 * 6);
    auto push = [&](const V& v) { out.insert(out.end(), {v.x, v.y, v.z, v.nx, v.ny, v.nz}); };
    for (int i = 0; i < stations; ++i) {
        for (int j = 0; j < sides; ++j) {
            const V& a = ring[(size_t)i * (sides + 1) + j];
            const V& b = ring[(size_t)(i + 1) * (sides + 1) + j];
            const V& c = ring[(size_t)(i + 1) * (sides + 1) + j + 1];
            const V& d = ring[(size_t)i * (sides + 1) + j + 1];
            push(a); push(b); push(c);
            push(a); push(c); push(d);
        }
    }
    return out;
}

// Unit cube (seat, floor, chair legs).
static std::vector<float> makeCube() {
    struct Face { float n[3], u[3], w[3]; };
    const Face faces[6] = {
        {{ 1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{ 0, 1, 0}, {1, 0, 0}, {0, 0, 1}}, {{ 0,-1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{ 0, 0, 1}, {1, 0, 0}, {0, 1, 0}}, {{ 0, 0,-1}, {1, 0, 0}, {0, 1, 0}},
    };
    std::vector<float> v;
    v.reserve(36 * 6);
    auto corner = [&](const Face& f, float su, float sw) {
        for (int k = 0; k < 3; ++k) v.push_back(0.5f * f.n[k] + 0.5f * su * f.u[k] + 0.5f * sw * f.w[k]);
        for (int k = 0; k < 3; ++k) v.push_back(f.n[k]);
    };
    for (const Face& f : faces) {
        corner(f, -1, -1); corner(f, 1, -1); corner(f, 1, 1);
        corner(f, -1, -1); corner(f, 1, 1); corner(f, -1, 1);
    }
    return v;
}
// ---- mesh (end) ----

struct Mesh {
    GLuint vao = 0, vbo = 0;
    GLsizei count = 0;
};

static Mesh uploadMesh(const std::vector<float>& data) {
    Mesh m;
    m.count = (GLsizei)(data.size() / 6);
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    return m;
}

// ------------------------------------------------------------- callbacks ---

static void framebufferSizeCallback(GLFWwindow*, int w, int h) {
    g_fbWidth = std::max(w, 1);
    g_fbHeight = std::max(h, 1);
    glViewport(0, 0, w, h);
}

static void keyCallback(GLFWwindow* win, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(win, GLFW_TRUE);
    if (key == GLFW_KEY_SPACE) g_autoSweep = !g_autoSweep;
    if (key == GLFW_KEY_R) { g_leg.kneeDeg = 0.0f; g_camYawDeg = 0.0f; }
    if (key == GLFW_KEY_Z) g_zeroRequested = true;
}

// ----------------------------------------------------------------- main ----

int main(int argc, char** argv) {
    parseArgs(argc, argv);

    std::thread liveThread(liveReaderThread, g_piCmd);

    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        g_running = false;
        liveThread.join();
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // 4x MSAA: smooth edges
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(g_fbWidth, g_fbHeight, "Brace Tracker", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        g_running = false;
        liveThread.join();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "gladLoadGLLoader failed\n");
        g_running = false;
        liveThread.join();
        return 1;
    }
    glfwGetFramebufferSize(window, &g_fbWidth, &g_fbHeight);
    glViewport(0, 0, g_fbWidth, g_fbHeight);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    GLuint program = makeProgram();
    GLint locModel = glGetUniformLocation(program, "uModel");
    GLint locViewProj = glGetUniformLocation(program, "uViewProj");
    GLint locViewPos = glGetUniformLocation(program, "uViewPos");
    GLint locColor = glGetUniformLocation(program, "uColor");
    GLint locSpec = glGetUniformLocation(program, "uSpec");
    GLint locShin = glGetUniformLocation(program, "uShininess");

    const Mesh thighMesh = uploadMesh(makeLathe(thighProfile));
    const Mesh shinMesh = uploadMesh(makeLathe(shinProfile));
    const Mesh ballMesh = uploadMesh(makeLathe(ballProfile, 32, 40));  // knee ball, foot
    const Mesh boxMesh = uploadMesh(makeCube());                      // seat, floor, chair legs

    double last = glfwGetTime();
    double sweepT = 0.0;

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = (float)(now - last);
        last = now;

        glfwPollEvents();

        auto held = [&](int key) { return glfwGetKey(window, key) == GLFW_PRESS; };
        if (held(GLFW_KEY_LEFT))  g_camYawDeg -= 60.0f * dt;
        if (held(GLFW_KEY_RIGHT)) g_camYawDeg += 60.0f * dt;
        if (held(GLFW_KEY_Q)) g_leg.thighCircCm += 10.0f * dt;
        if (held(GLFW_KEY_A)) g_leg.thighCircCm -= 10.0f * dt;
        if (held(GLFW_KEY_W)) g_leg.calfCircCm += 10.0f * dt;
        if (held(GLFW_KEY_S)) g_leg.calfCircCm -= 10.0f * dt;
        if (held(GLFW_KEY_EQUAL)) g_leg.scale += 0.5f * dt;
        if (held(GLFW_KEY_MINUS)) g_leg.scale -= 0.5f * dt;

        // ---- choose the angle source ----
        const long long lastLive = g_lastLiveMs.load();
        const bool everLive = lastLive >= 0;
        const bool liveFresh = everLive && (nowMs() - lastLive) <= 500;
        const char* source;
        float quality = -1.0f;
        if (liveFresh) {
            g_leg.kneeDeg = g_liveKnee.load();
            quality = g_liveQuality.load();
            g_autoSweep = false;  // real data takes over from the demo
            source = g_zeroing.load() ? "LIVE - ZEROING" : "LIVE";
        } else if (everLive && !g_autoSweep) {
            source = "SIGNAL LOST";  // hold the last angle
        } else if (g_autoSweep) {
            sweepT += dt;
            // Seated leg extension: ~90 deg flexion down to 0 (straight) and back, one rep per 4 s.
            g_leg.kneeDeg = 45.0f * (1.0f + std::cos((float)sweepT * 2.0f * kPi / 4.0f));
            source = "AUTO DEMO";
        } else {
            if (held(GLFW_KEY_UP))   g_leg.kneeDeg += 60.0f * dt;
            if (held(GLFW_KEY_DOWN)) g_leg.kneeDeg -= 60.0f * dt;
            source = "MANUAL";
        }
        clampLeg();

        char title[200];
        if (quality >= 0.0f) {
            std::snprintf(title, sizeof(title),
                          "Brace Tracker | knee %.1f deg | %s q=%.2f | thigh circ %.0f cm | calf circ %.0f cm | size x%.2f",
                          g_leg.kneeDeg, source, quality, g_leg.thighCircCm, g_leg.calfCircCm, g_leg.scale);
        } else {
            std::snprintf(title, sizeof(title),
                          "Brace Tracker | knee %.1f deg | %s | thigh circ %.0f cm | calf circ %.0f cm | size x%.2f",
                          g_leg.kneeDeg, source, g_leg.thighCircCm, g_leg.calfCircCm, g_leg.scale);
        }
        glfwSetWindowTitle(window, title);

        // ---- render ----
        glClearColor(0.07f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program);

        // Camera orbits a point between the hip and the foot, slightly above.
        float aspect = (float)g_fbWidth / (float)g_fbHeight;
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 60.0f);
        float yaw = glm::radians(g_camYawDeg);
        const glm::vec3 target(-0.45f, -0.95f, 0.0f);
        const glm::vec3 eye = target + glm::vec3(5.4f * std::sin(yaw), 1.6f, 5.4f * std::cos(yaw));
        glm::mat4 view = glm::lookAt(eye, target, glm::vec3(0, 1, 0));
        glm::mat4 viewProj = proj * view;
        glUniformMatrix4fv(locViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
        glUniform3fv(locViewPos, 1, glm::value_ptr(eye));

        auto draw = [&](const Mesh& mesh, const glm::mat4& model, const glm::vec3& color,
                        float spec = 0.15f, float shininess = 24.0f) {
            glUniformMatrix4fv(locModel, 1, GL_FALSE, glm::value_ptr(model));
            glUniform3fv(locColor, 1, glm::value_ptr(color));
            glUniform1f(locSpec, spec);
            glUniform1f(locShin, shininess);
            glBindVertexArray(mesh.vao);
            glDrawArrays(GL_TRIANGLES, 0, mesh.count);
        };

        // Every size below is derived from the four measurements (cm) and the overall
        // scale, so changing any one of them resizes everything that depends on it.
        auto cm = [](float v) { return v * kUnitsPerCm * g_leg.scale; };
        const glm::mat4 I(1.0f);
        const float thighLenCm = g_leg.thighLenCm, shinLenCm = g_leg.shinLenCm;
        const float thighDcm = diameterCm(g_leg.thighCircCm);  // widest diameters
        const float calfDcm = diameterCm(g_leg.calfCircCm);
        const float footLenCm = 0.62f * shinLenCm;              // foot proportions follow the leg
        const float footThickCm = 0.68f * calfDcm;
        const float footWidthCm = 0.85f * calfDcm;
        const float thighLen = cm(thighLenCm), shinLen = cm(shinLenCm);
        const float thighD = cm(thighDcm), calfD = cm(calfDcm);

        const glm::vec3 skin(0.86f, 0.66f, 0.54f);
        const glm::vec3 metal(0.62f, 0.64f, 0.68f);
        const glm::vec3 shoe(0.84f, 0.84f, 0.86f);

        // Knee is the origin. Thigh runs from the hip at -X to the knee.
        // At 0 deg the shin continues straight along +X; flexion swings it downward
        // (90 deg = shin hanging straight down, like sitting on a chair edge).
        const glm::mat4 shinFrame = glm::rotate(I, -glm::radians(g_leg.kneeDeg), glm::vec3(0, 0, 1));

        // -- scene: seat + floor, so the leg has context (all sizes follow the leg) --
        if (kShowSeatAndFloor) {
            const float seatTop = -0.5f * thighD;
            const float seatThick = cm(0.23f * thighDcm);
            // The seat ends behind the knee, far enough back that the calf swings past its front
            // edge without touching it, for every angle up to kMaxKneeDeg.
            const float rCalf = cm(0.5f * calfDcm);                       // widest calf radius
            const float seatDepth = -(seatTop - seatThick);               // knee level -> seat underside
            const float th = glm::radians(kMaxKneeDeg);
            const float sClear = (seatDepth - rCalf * std::cos(th)) / std::sin(th);
            const float seatX0 = -(thighLen + cm(0.18f * thighLenCm));
            const float seatX1 = std::min(sClear * std::cos(th) - rCalf * std::sin(th), -rCalf) - cm(1.5f);
            const float floorY = seatTop - cm(shinLenCm + 1.5f * footThickCm);
            const float legW = cm(0.08f * thighLenCm), legZ = cm(0.40f * thighLenCm);
            const float legH = (seatTop - seatThick) - floorY;
            const glm::vec3 seatColor(0.30f, 0.31f, 0.35f);
            draw(boxMesh, glm::scale(glm::translate(I, glm::vec3(0.5f * (seatX0 + seatX1), seatTop - 0.5f * seatThick, 0)),
                                     glm::vec3(seatX1 - seatX0, seatThick, cm(thighLenCm))), seatColor, 0.05f, 16.0f);
            for (float lx : {seatX0 + legW, seatX1 - legW})
                for (float lz : {-legZ, legZ})
                    draw(boxMesh, glm::scale(glm::translate(I, glm::vec3(lx, floorY + 0.5f * legH, lz)),
                                             glm::vec3(legW, legH, legW)), metal, 0.3f, 40.0f);
            draw(boxMesh, glm::scale(glm::translate(I, glm::vec3(0, floorY - cm(1.0f), 0)),
                                     glm::vec3(cm(500.0f), cm(2.0f), cm(500.0f))), glm::vec3(0.16f, 0.17f, 0.20f), 0.04f, 12.0f);
        }

        // -- thigh and shin (tapered, sized from the circumferences) --
        draw(thighMesh, glm::scale(glm::translate(I, glm::vec3(-thighLen * 0.5f, 0, 0)),
                                   glm::vec3(thighLen, thighD, thighD)), skin, 0.12f, 18.0f);
        draw(shinMesh, glm::scale(glm::translate(shinFrame, glm::vec3(shinLen * 0.5f, 0, 0)),
                                  glm::vec3(shinLen, calfD, calfD)), skin, 0.12f, 18.0f);

        // -- knee ball hides the seam between the two segments --
        const float kneeR = 0.5f * std::max(thighD * thighProfile(1.0f), calfD * shinProfile(0.0f)) * 1.02f;
        draw(ballMesh, glm::scale(I, glm::vec3(2.0f * kneeR)), skin, 0.12f, 18.0f);

        // -- foot (a shoe): points forward, i.e. along the shin frame's +Y --
        draw(ballMesh, glm::scale(glm::translate(shinFrame, glm::vec3(shinLen + cm(0.30f * calfDcm), cm(0.23f * footLenCm), 0)),
                                  glm::vec3(cm(footThickCm), cm(footLenCm), cm(footWidthCm))), shoe, 0.35f, 48.0f);

        glfwSwapBuffers(window);
    }

    g_running = false;
    liveThread.join();  // the helper exits when its pipe closes
    for (const Mesh* m : {&thighMesh, &shinMesh, &ballMesh, &boxMesh}) {
        glDeleteBuffers(1, &m->vbo);
        glDeleteVertexArrays(1, &m->vao);
    }
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}