const std = @import("std");

const StrideWalker = @This();

ptr: [*]f32,
end: [*]f32,
rowEnd: [*]f32,
widthMinusOne: u32,
paddingPlusOne: u32,

pub fn init(ptr: [*]f32, width: u32, stride: u32, height: u32) StrideWalker {
    const widthMinusOne = width - 1;
    const paddingPlusOne = stride - widthMinusOne;
    return .{
        .ptr = ptr - 1,
        .end = ptr + stride * height - paddingPlusOne,
        .rowEnd = ptr + widthMinusOne,
        .widthMinusOne = widthMinusOne,
        .paddingPlusOne = paddingPlusOne,
    };
}

pub fn next(w: *StrideWalker) bool {
    if (@intFromPtr(w.ptr) < @intFromPtr(w.rowEnd)) {
        w.ptr += 1;
        return true;
    }
    if (@intFromPtr(w.ptr) < @intFromPtr(w.end)) {
        w.ptr += w.paddingPlusOne;
        w.rowEnd = w.ptr + w.widthMinusOne;
        return true;
    }
    return false;
}

pub fn testingApproxEq(w: *StrideWalker, expected: []const f32) !void {
    for (expected) |v| {
        try std.testing.expect(w.next());
        try std.testing.expectApproxEqAbs(v, w.ptr[0], 0.0001);
    }
    try std.testing.expect(!w.next());
}

pub const Row = struct {
    ptr: [*]f32,
    end: [*]f32,
    width: u32,
    stride: u32,

    pub fn init(ptr: [*]f32, width: u32, stride: u32, height: u32) Row {
        return .{
            .ptr = ptr,
            .end = ptr + stride * height,
            .width = width,
            .stride = stride,
        };
    }
    pub fn next(row: *Row) ?[]f32 {
        if (@intFromPtr(row.ptr) >= @intFromPtr(row.end)) {
            return null;
        }
        const out = row.ptr[0..row.width];
        row.ptr += row.stride;
        return out;
    }
};

test "stridewalker" {
    var buf = [_]f32{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    var walker = StrideWalker.init(&buf, 3, 4, 2);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(0.0, walker.ptr[0]);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(1.0, walker.ptr[0]);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(2.0, walker.ptr[0]);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(4.0, walker.ptr[0]);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(5.0, walker.ptr[0]);
    try std.testing.expect(walker.next());
    try std.testing.expectEqual(6.0, walker.ptr[0]);
    try std.testing.expect(!walker.next());
}
