const std = @import("std");

fn StrideWalkerBase(comptime isConst: bool) type {
    const Ptr = if (isConst) [*]const f32 else [*]f32;

    return struct {
        ptr: Ptr,
        end: Ptr,
        rowEnd: Ptr,
        widthMinusOne: u32,
        paddingPlusOne: u32,

        pub fn init(ptr: Ptr, width: u32, stride: u32, height: u32) @This() {
            std.debug.assert(width <= stride);
            if (height == 0 or width == 0) {
                return .{
                    .ptr = ptr,
                    .end = ptr,
                    .rowEnd = ptr,
                    .widthMinusOne = 0,
                    .paddingPlusOne = 0,
                };
            }
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

        pub fn next(w: *@This()) bool {
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

        pub fn testingApproxEq(w: *@This(), expected: []const f32) !void {
            for (expected) |v| {
                try std.testing.expect(w.next());
                try std.testing.expectApproxEqAbs(v, w.ptr[0], 0.0001);
            }
            try std.testing.expect(!w.next());
        }
    };
}

pub const StrideWalker = StrideWalkerBase(true);
pub const StrideWalkerMut = StrideWalkerBase(false);

fn StrideWalkerRowBase(comptime isConst: bool) type {
    const Ptr = if (isConst) [*]const f32 else [*]f32;
    const Slice = if (isConst) []const f32 else []f32;
    return struct {
        ptr: Ptr,
        end: Ptr,
        width: u32,
        stride: u32,

        pub fn init(ptr: Ptr, width: u32, stride: u32, height: u32) @This() {
            return .{
                .ptr = ptr,
                .end = ptr + stride * height,
                .width = width,
                .stride = stride,
            };
        }
        pub fn next(row: *@This()) ?Slice {
            if (@intFromPtr(row.ptr) >= @intFromPtr(row.end)) {
                return null;
            }
            const out = row.ptr[0..row.width];
            row.ptr += row.stride;
            return out;
        }
    };
}

pub const StrideWalkerRow = StrideWalkerRowBase(true);
pub const StrideWalkerRowMut = StrideWalkerRowBase(false);

test "stridewalker zero" {
    var buf = [_]f32{};
    var walker = StrideWalker.init(&buf, 0, 0, 2);
    try std.testing.expect(!walker.next());
    walker = StrideWalker.init(&buf, 2, 2, 0);
    try std.testing.expect(!walker.next());
}

test "stridewalker standard" {
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
