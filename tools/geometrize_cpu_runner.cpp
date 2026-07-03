#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "geometrize/bitmap/bitmap.h"
#include "geometrize/commonutil.h"
#include "geometrize/exporter/bitmapexporter.h"
#include "geometrize/exporter/shapejsonexporter.h"
#include "geometrize/rasterizer/rasterizer.h"
#include "geometrize/runner/imagerunner.h"
#include "geometrize/runner/imagerunneroptions.h"
#include "geometrize/shape/shapetypes.h"
#include "geometrize/shape/triangle.h"
#include "geometrize/shaperesult.h"

namespace {

constexpr float kPi = 3.14159265358979323846f;

class IsoscelesTriangleShape : public geometrize::Triangle {
public:
    float cx = 0.0f;
    float cy = 0.0f;
    float half_base = 1.0f;
    float height = 1.0f;
    float theta = 0.0f;

    std::shared_ptr<geometrize::Shape> clone() const override {
        auto triangle = std::make_shared<IsoscelesTriangleShape>(*this);
        triangle->setup = setup;
        triangle->mutate = mutate;
        triangle->rasterize = rasterize;
        return triangle;
    }

    geometrize::ShapeTypes getType() const override {
        return geometrize::ShapeTypes::TRIANGLE;
    }

    void update_vertices() {
        const float ct = std::cos(theta);
        const float st = std::sin(theta);
        const auto rotate_x = [&](const float x, const float y) {
            return cx + ct * x - st * y;
        };
        const auto rotate_y = [&](const float x, const float y) {
            return cy + st * x + ct * y;
        };

        const float apex_x = 0.0f;
        const float apex_y = -height * 0.5f;
        const float left_x = -half_base;
        const float left_y = height * 0.5f;
        const float right_x = half_base;
        const float right_y = height * 0.5f;
        m_x1 = rotate_x(apex_x, apex_y);
        m_y1 = rotate_y(apex_x, apex_y);
        m_x2 = rotate_x(left_x, left_y);
        m_y2 = rotate_y(left_x, left_y);
        m_x3 = rotate_x(right_x, right_y);
        m_y3 = rotate_y(right_x, right_y);
    }
};

float random_float(float low, float high) {
    if(high <= low) {
        return low;
    }
    const int scaled = geometrize::commonutil::randomRange(0, 1000000);
    return low + (high - low) * (static_cast<float>(scaled) / 1000000.0f);
}

std::function<std::shared_ptr<geometrize::Shape>()> make_isosceles_creator(std::uint32_t width, std::uint32_t height) {
    const float min_half_base = std::max(1.0f, static_cast<float>(width) / 256.0f);
    const float max_half_base = std::max(min_half_base, static_cast<float>(width) * 0.125f);
    const float min_height = std::max(1.0f, static_cast<float>(height) / 256.0f);
    const float max_height = std::max(min_height, static_cast<float>(height) * 0.25f);
    const float center_step_x = static_cast<float>(width) * 0.125f;
    const float center_step_y = static_cast<float>(height) * 0.125f;
    const float half_base_step = static_cast<float>(width) * 0.0625f;
    const float height_step = static_cast<float>(height) * 0.0625f;
    const float angle_step = 16.0f * kPi / 180.0f;

    return [=]() -> std::shared_ptr<geometrize::Shape> {
        auto shape = std::make_shared<IsoscelesTriangleShape>();
        shape->setup = [=](geometrize::Shape& raw) {
            auto& tri = static_cast<IsoscelesTriangleShape&>(raw);
            tri.cx = random_float(0.5f, static_cast<float>(width) - 0.5f);
            tri.cy = random_float(0.5f, static_cast<float>(height) - 0.5f);
            tri.half_base = random_float(min_half_base, max_half_base);
            tri.height = random_float(min_height, max_height);
            tri.theta = random_float(0.0f, 2.0f * kPi);
            tri.update_vertices();
        };
        shape->mutate = [=](geometrize::Shape& raw) {
            auto& tri = static_cast<IsoscelesTriangleShape&>(raw);
            const int choice = geometrize::commonutil::randomRange(0, 3);
            if(choice == 0) {
                tri.cx += random_float(-center_step_x, center_step_x);
                tri.cy += random_float(-center_step_y, center_step_y);
                tri.cx = std::clamp(tri.cx, 0.5f, static_cast<float>(width) - 0.5f);
                tri.cy = std::clamp(tri.cy, 0.5f, static_cast<float>(height) - 0.5f);
            } else if(choice == 1) {
                tri.half_base += random_float(-half_base_step, half_base_step);
                tri.half_base = std::clamp(tri.half_base, min_half_base, max_half_base);
            } else if(choice == 2) {
                tri.height += random_float(-height_step, height_step);
                tri.height = std::clamp(tri.height, min_height, max_height);
            } else {
                tri.theta += random_float(-angle_step, angle_step);
                tri.theta = std::fmod(tri.theta, 2.0f * kPi);
                if(tri.theta < 0.0f) {
                    tri.theta += 2.0f * kPi;
                }
            }
            tri.update_vertices();
        };
        shape->rasterize = [=](const geometrize::Shape& raw) {
            return geometrize::rasterize(static_cast<const geometrize::Triangle&>(raw), 0, 0, static_cast<std::int32_t>(width), static_cast<std::int32_t>(height));
        };
        shape->setup(*shape);
        return shape;
    };
}

std::uint16_t read_u16(const std::vector<std::uint8_t>& bytes, const std::size_t offset) {
    return static_cast<std::uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& bytes, const std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16) | (bytes[offset + 3] << 24));
}

std::int32_t read_i32(const std::vector<std::uint8_t>& bytes, const std::size_t offset) {
    return static_cast<std::int32_t>(read_u32(bytes, offset));
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if(!input) {
        throw std::runtime_error("failed to open input: " + path);
    }
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

void write_text_file(const std::string& path, const std::string& data) {
    std::ofstream output(path, std::ios::binary);
    if(!output) {
        throw std::runtime_error("failed to open output: " + path);
    }
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
}

geometrize::Bitmap load_bmp_rgba(const std::string& path) {
    const std::vector<std::uint8_t> bytes = read_file(path);
    if(bytes.size() < 54 || bytes[0] != 'B' || bytes[1] != 'M') {
        throw std::runtime_error("expected an uncompressed BMP file");
    }
    const std::uint32_t pixel_offset = read_u32(bytes, 10);
    const std::uint32_t dib_size = read_u32(bytes, 14);
    if(dib_size < 40) {
        throw std::runtime_error("unsupported BMP DIB header");
    }
    const std::int32_t width_i = read_i32(bytes, 18);
    const std::int32_t height_i = read_i32(bytes, 22);
    const std::uint16_t planes = read_u16(bytes, 26);
    const std::uint16_t bits_per_pixel = read_u16(bytes, 28);
    const std::uint32_t compression = read_u32(bytes, 30);
    if(width_i <= 0 || height_i == 0 || planes != 1 || compression != 0 || (bits_per_pixel != 24 && bits_per_pixel != 32)) {
        throw std::runtime_error("only uncompressed 24-bit or 32-bit BMP files are supported");
    }

    const std::uint32_t width = static_cast<std::uint32_t>(width_i);
    const std::uint32_t height = static_cast<std::uint32_t>(height_i < 0 ? -height_i : height_i);
    const bool top_down = height_i < 0;
    const std::uint32_t src_stride = ((width * bits_per_pixel + 31U) / 32U) * 4U;
    std::vector<std::uint8_t> rgba(width * height * 4U, 255U);
    for(std::uint32_t y = 0; y < height; ++y) {
        const std::uint32_t src_y = top_down ? y : (height - 1U - y);
        const std::size_t row = static_cast<std::size_t>(pixel_offset) + static_cast<std::size_t>(src_y) * src_stride;
        if(row + src_stride > bytes.size()) {
            throw std::runtime_error("truncated BMP pixel data");
        }
        for(std::uint32_t x = 0; x < width; ++x) {
            const std::size_t src = row + static_cast<std::size_t>(x) * (bits_per_pixel / 8U);
            const std::size_t dst = (static_cast<std::size_t>(y) * width + x) * 4U;
            rgba[dst + 0] = bytes[src + 2];
            rgba[dst + 1] = bytes[src + 1];
            rgba[dst + 2] = bytes[src + 0];
            rgba[dst + 3] = bits_per_pixel == 32 ? bytes[src + 3] : 255U;
        }
    }
    return geometrize::Bitmap(width, height, rgba);
}

struct Args {
    std::string input;
    std::string output_bmp = "out/geometrize_cpu/final.bmp";
    std::string output_json = "out/geometrize_cpu/shapes.json";
    std::uint32_t num_triangles = 300;
    std::uint32_t candidate_count = 2048;
    std::uint32_t max_shape_mutations = 2000;
    std::uint32_t max_threads = 0;
    std::uint32_t seed = 1;
    std::uint8_t alpha = 255;
    std::string shape = "isosceles";
};

Args parse_args(int argc, char** argv) {
    Args args;
    for(int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const auto require_value = [&](const std::string& name) -> std::string {
            if(i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };
        if(key == "--input") {
            args.input = require_value(key);
        } else if(key == "--output-bmp") {
            args.output_bmp = require_value(key);
        } else if(key == "--output-json") {
            args.output_json = require_value(key);
        } else if(key == "--num-triangles") {
            args.num_triangles = static_cast<std::uint32_t>(std::stoul(require_value(key)));
        } else if(key == "--candidate-count") {
            args.candidate_count = static_cast<std::uint32_t>(std::stoul(require_value(key)));
        } else if(key == "--max-shape-mutations") {
            args.max_shape_mutations = static_cast<std::uint32_t>(std::stoul(require_value(key)));
        } else if(key == "--max-threads") {
            args.max_threads = static_cast<std::uint32_t>(std::stoul(require_value(key)));
        } else if(key == "--seed") {
            args.seed = static_cast<std::uint32_t>(std::stoul(require_value(key)));
        } else if(key == "--alpha") {
            args.alpha = static_cast<std::uint8_t>(std::clamp(std::stoi(require_value(key)), 0, 255));
        } else if(key == "--shape") {
            args.shape = require_value(key);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if(args.input.empty()) {
        throw std::runtime_error("--input is required");
    }
    if(args.shape != "isosceles" && args.shape != "triangle") {
        throw std::runtime_error("--shape must be either isosceles or triangle");
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        geometrize::Bitmap target = load_bmp_rgba(args.input);
        geometrize::ImageRunner runner(target);
        geometrize::ImageRunnerOptions options;
        options.shapeTypes = geometrize::ShapeTypes::TRIANGLE;
        options.alpha = args.alpha;
        options.shapeCount = args.candidate_count;
        options.maxShapeMutations = args.max_shape_mutations;
        options.maxThreads = args.max_threads;
        options.seed = args.seed;
        const std::function<std::shared_ptr<geometrize::Shape>()> shape_creator =
            args.shape == "isosceles" ? make_isosceles_creator(target.getWidth(), target.getHeight()) : nullptr;

        std::vector<geometrize::ShapeResult> shapes;
        shapes.reserve(args.num_triangles);
        const auto start = std::chrono::steady_clock::now();
        for(std::uint32_t i = 0; i < args.num_triangles; ++i) {
            options.seed = args.seed + i;
            std::vector<geometrize::ShapeResult> result = runner.step(options, shape_creator);
            if(result.empty()) {
                std::cout << "stopped at triangle " << i << " because no improving shape was accepted\n";
                break;
            }
            for(const geometrize::ShapeResult& shape : result) {
                shapes.push_back(shape);
            }
            if(i == 0 || (i + 1U) % 10U == 0 || i + 1U == args.num_triangles) {
                std::cout << "[triangle " << (i + 1U) << "/" << args.num_triangles << "] score=" << result.back().score << "\n";
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();

        write_text_file(args.output_bmp, geometrize::exporter::exportBMP(runner.getCurrent()));
        write_text_file(args.output_json, geometrize::exporter::exportShapeJson(shapes));
        std::cout << "triangles=" << shapes.size() << "\n";
        std::cout << "shape=" << args.shape << "\n";
        std::cout << "seconds=" << seconds << "\n";
        std::cout << "saved_bmp=" << args.output_bmp << "\n";
        std::cout << "saved_json=" << args.output_json << "\n";
        return 0;
    } catch(const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
