const std = @import("std");

const BuildContext = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    mininet: *std.Build.Module,

    fn addExample(ctx: *BuildContext, name: []const u8, desc: []const u8, path: []const u8) void {
        const module = ctx.b.createModule(.{
            .target = ctx.target,
            .optimize = ctx.optimize,
            .root_source_file = ctx.b.path(path),
        });
        module.addImport("mininet", ctx.mininet);
        const compile = ctx.b.addExecutable(.{
            .name = name,
            .root_module = module,
        });
        ctx.b.installArtifact(compile);
        ctx.b.step(name, desc).dependOn(&ctx.b.addRunArtifact(compile).step);
    }
};

pub fn build(b: *std.Build) !void {
    var ctx = BuildContext{
        .b = b,
        .target = b.standardTargetOptions(.{}),
        .optimize = b.standardOptimizeOption(.{}),
        .mininet = undefined,
    };
    ctx.mininet = b.addModule("mininet", .{
        .target = ctx.target,
        .optimize = ctx.optimize,
        .root_source_file = b.path("src/mininet.zig"),
    });

    // example
    ctx.addExample("example-basic", "Run basic example", "examples/basic.zig");
    ctx.addExample("example-load", "Run load example", "examples/load.zig");
    ctx.addExample("example-load-raw", "Run load-raw example", "examples/load_raw.zig");

    // test
    const test_filters = b.option(
        []const []const u8,
        "test-filter",
        "Run only matching tests",
    ) orelse &.{};
    const emit_bin: []const u8 = b.option(
        []const u8,
        "emit-bin",
        "Run only matching tests",
    ) orelse &.{};
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .target = ctx.target,
            .optimize = ctx.optimize,
            .root_source_file = b.path("src/tests.zig"),
        }),
        .filters = test_filters,
    });
    b.step("test", "Run unit tests").dependOn(&b.addRunArtifact(tests).step);

    if (b.build_root.path) |bpath| {
        const rpath = try std.fs.path.relative(b.allocator, bpath, null, bpath, emit_bin);
        b.step("install-test", "Build unit tests").dependOn(&b.addInstallArtifact(tests, .{
            .dest_dir = .default,
            .dest_sub_path = rpath,
        }).step);
    }
}
