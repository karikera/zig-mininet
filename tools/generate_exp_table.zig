const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const file = try std.Io.Dir.cwd().createFile(init.io, "src/generated_exp_table.zig", .{});
    defer file.close(init.io);

    var buf: [8192]u8 = undefined;
    var writer = file.writer(init.io, &buf);
    const w = &writer.interface;

    try w.writeAll("pub const expIntOffset = 6;\n");
    try w.writeAll("pub const expIntShift = 8;\n");
    try w.writeAll("pub const expIntTable: [11]u32 = .{\n");
    for (0..11) |n| {
        const f: f64 = @floatFromInt(n);
        const res = std.math.round(std.math.exp(f + 6.0) * 0x100); // 8 bits
        const ires: u32 = @intFromFloat(res);
        try w.print("    {},\n", .{ires});
    }
    try w.writeAll("};\n");
    try w.writeAll("pub const expMinusIntOffset = 5;\n");
    try w.writeAll("pub const expMinusIntShift = 24;\n");
    try w.writeAll("pub const expMinusIntTable: [14]u32 = .{\n");
    for (0..14) |n| {
        const f: f64 = @floatFromInt(n);
        const res = std.math.round(std.math.exp(5.0 - f) * 0x1000000); // 24 bits
        const ires: u32 = @intFromFloat(res);
        try w.print("    {},\n", .{ires});
    }
    try w.writeAll("};\n");
    try w.writeAll("pub const expHighMShift = 16;\n");
    try w.writeAll("pub const expHighMTable: [16]u32 = .{\n");
    for (0..16) |n| {
        const f: f64 = @floatFromInt(n);
        const res = std.math.round(std.math.exp(f / 16.0) * 0x10000); // 16 bits
        const ires: u32 = @intFromFloat(res);
        try w.print("    {},\n", .{ires});
    }
    try w.writeAll("};\n");
    try w.writeAll("pub const expLowMShift = 15;\n");
    try w.writeAll("pub const expLowMTable: [16]u32 = .{\n");
    for (0..16) |n| {
        const f: f64 = @floatFromInt(n);
        const res = std.math.round(std.math.exp(f / 256.0) * 0x8000); // 15 bits
        const ires: u32 = @intFromFloat(res);
        try w.print("    {},\n", .{ires});
    }
    try w.writeAll("};\n");
    try w.flush();
}
