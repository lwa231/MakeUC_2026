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
//   LEFT / RIGHT  orbit camera
//   SPACE         toggle the auto demo sweep
//   R             reset angle and camera
//   ESC           quit
//
// Options: --pi-cmd "command"   (default: "python fake_pi.py", or python3 off Windows)

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

struct LegState {
    float kneeDeg = 0.0f;         // 0 = full extension, flexion positive

    // ---- body measurements in cm: PUT THE PATIENT'S VALUES HERE ----
    float thighCircCm = 55.0f;    // circumference around the widest part of the thigh
    float calfCircCm = 37.0f;     // circumference around the widest part of the calf
    float thighLenCm = 45.0f;     // hip -> knee
    float shinLenCm = 42.0f;      // knee -> ankle
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
    g_leg.kneeDeg = std::clamp(g_leg.kneeDeg, 0.0f, 140.0f);
    g_leg.thighCircCm = std::clamp(g_leg.thighCircCm, 20.0f, 100.0f);
    g_leg.calfCircCm = std::clamp(g_leg.calfCircCm, 15.0f, 80.0f);
    g_leg.thighLenCm = std::clamp(g_leg.thighLenCm, 20.0f, 80.0f);
    g_leg.shinLenCm = std::clamp(g_leg.shinLenCm, 20.0f, 80.0f);
}

static void parseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        float* target = nullptr;
        if (!std::strcmp(argv[i], "--thigh-circ")) target = &g_leg.thighCircCm;
        else if (!std::strcmp(argv[i], "--calf-circ")) target = &g_leg.calfCircCm;
        else if (!std::strcmp(argv[i], "--thigh-len")) target = &g_leg.thighLenCm;
        else if (!std::strcmp(argv[i], "--shin-len")) target = &g_leg.shinLenCm;

        if (target && i + 1 < argc) {
            *target = (float)std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--pi-cmd") && i + 1 < argc) {
            g_piCmd = argv[++i];
        } else {
            std::fprintf(stderr,
                         "Unknown or incomplete option '%s'\n"
                         "Usage: viewer [--thigh-circ cm] [--calf-circ cm] [--thigh-len cm] [--shin-len cm]\n"
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
void main() {
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    gl_Position = uViewProj * uModel * vec4(aPos, 1.0);
}
)";

static const char* kFragSrc = R"(#version 330 core
in vec3 vNormal;
uniform vec3 uColor;
out vec4 FragColor;
void main() {
    vec3 n = normalize(vNormal);
    vec3 lightDir = normalize(vec3(0.4, 0.8, 0.6));
    float diff = max(dot(n, lightDir), 0.0);
    FragColor = vec4(uColor * (0.25 + 0.75 * diff), 1.0);
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

// Unit cylinder: axis along X from -0.5 to +0.5, diameter 1, with flat caps.
// Triangle list, each vertex (x y z nx ny nz).
static std::vector<float> makeCylinder(int sides = 32) {
    std::vector<float> v;
    v.reserve(sides * 12 * 6);
    auto push = [&](float x, float y, float z, float nx, float ny, float nz) {
        v.insert(v.end(), {x, y, z, nx, ny, nz});
    };
    for (int i = 0; i < sides; ++i) {
        float a0 = 2.0f * kPi * i / sides;
        float a1 = 2.0f * kPi * (i + 1) / sides;
        float y0 = 0.5f * std::cos(a0), z0 = 0.5f * std::sin(a0);
        float y1 = 0.5f * std::cos(a1), z1 = 0.5f * std::sin(a1);

        // side wall (two triangles)
        push(-0.5f, y0, z0, 0, std::cos(a0), std::sin(a0));
        push( 0.5f, y0, z0, 0, std::cos(a0), std::sin(a0));
        push( 0.5f, y1, z1, 0, std::cos(a1), std::sin(a1));
        push(-0.5f, y0, z0, 0, std::cos(a0), std::sin(a0));
        push( 0.5f, y1, z1, 0, std::cos(a1), std::sin(a1));
        push(-0.5f, y1, z1, 0, std::cos(a1), std::sin(a1));

        // end caps
        push( 0.5f, 0, 0, 1, 0, 0); push(0.5f, y0, z0, 1, 0, 0); push(0.5f, y1, z1, 1, 0, 0);
        push(-0.5f, 0, 0, -1, 0, 0); push(-0.5f, y1, z1, -1, 0, 0); push(-0.5f, y0, z0, -1, 0, 0);
    }
    return v;
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

    GLuint program = makeProgram();
    GLint locModel = glGetUniformLocation(program, "uModel");
    GLint locViewProj = glGetUniformLocation(program, "uViewProj");
    GLint locColor = glGetUniformLocation(program, "uColor");

    std::vector<float> mesh = makeCylinder();
    const GLsizei vertexCount = (GLsizei)(mesh.size() / 6);
    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, mesh.size() * sizeof(float), mesh.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

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
                          "Brace Tracker | knee %.1f deg | %s q=%.2f | thigh circ %.0f cm | calf circ %.0f cm",
                          g_leg.kneeDeg, source, quality, g_leg.thighCircCm, g_leg.calfCircCm);
        } else {
            std::snprintf(title, sizeof(title),
                          "Brace Tracker | knee %.1f deg | %s | thigh circ %.0f cm | calf circ %.0f cm",
                          g_leg.kneeDeg, source, g_leg.thighCircCm, g_leg.calfCircCm);
        }
        glfwSetWindowTitle(window, title);

        // ---- render ----
        glClearColor(0.09f, 0.10f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program);

        float aspect = (float)g_fbWidth / (float)g_fbHeight;
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 50.0f);
        float yaw = glm::radians(g_camYawDeg);
        glm::vec3 eye(5.0f * std::sin(yaw), 0.3f, 5.0f * std::cos(yaw));  // side-on view down -Z at yaw 0
        glm::mat4 view = glm::lookAt(eye, glm::vec3(0.3f, -0.6f, 0.0f), glm::vec3(0, 1, 0));
        glm::mat4 viewProj = proj * view;
        glUniformMatrix4fv(locViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));

        glBindVertexArray(vao);

        auto draw = [&](const glm::mat4& model, const glm::vec3& color) {
            glUniformMatrix4fv(locModel, 1, GL_FALSE, glm::value_ptr(model));
            glUniform3fv(locColor, 1, glm::value_ptr(color));
            glDrawArrays(GL_TRIANGLES, 0, vertexCount);
        };

        const float thighLen = g_leg.thighLenCm * kUnitsPerCm;
        const float shinLen = g_leg.shinLenCm * kUnitsPerCm;
        const float thighD = diameterCm(g_leg.thighCircCm) * kUnitsPerCm;
        const float calfD = diameterCm(g_leg.calfCircCm) * kUnitsPerCm;

        // Knee is the origin. Thigh runs from the hip at -X to the knee.
        // At 0 deg the shin continues straight along +X; flexion swings it downward
        // (90 deg = shin hanging straight down, like sitting on a chair edge).
        glm::mat4 thigh = glm::translate(glm::mat4(1.0f), glm::vec3(-thighLen * 0.5f, 0, 0));
        thigh = glm::scale(thigh, glm::vec3(thighLen, thighD, thighD));
        draw(thigh, glm::vec3(0.30f, 0.60f, 0.95f));

        glm::mat4 shin = glm::rotate(glm::mat4(1.0f), -glm::radians(g_leg.kneeDeg), glm::vec3(0, 0, 1));
        shin = glm::translate(shin, glm::vec3(shinLen * 0.5f, 0, 0));
        shin = glm::scale(shin, glm::vec3(shinLen, calfD, calfD));
        draw(shin, glm::vec3(0.95f, 0.55f, 0.25f));

        // Hinge pin: a short cylinder along Z at the knee.
        const float jointD = 0.5f * (thighD + calfD);
        glm::mat4 joint = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 1, 0));
        joint = glm::scale(joint, glm::vec3(thighD * 1.15f, jointD, jointD));
        draw(joint, glm::vec3(0.90f, 0.90f, 0.90f));

        glfwSwapBuffers(window);
    }

    g_running = false;
    liveThread.join();  // the helper exits when its pipe closes
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}