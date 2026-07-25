# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: '/home/jjf190004/pyllm_dataset/pymunk/d42177d354ccd4d76259b02c9a4982cbfa9dd512030d0a860ddbcb9a13f4ddfa973ba70bfe969a06893ef07f3fa3e6687d710f594d77bee2437dde09faf88c15/space_debug_draw_options.py'
# Bytecode version: 3.15 (3666)
# Source timestamp: 2026-07-08 16:40:16 UTC (1783528816)

__docformat__ = 'reStructuredText'
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Optional, Sequence
if TYPE_CHECKING:
    from .shapes import Shape
    from types import TracebackType
from ._chipmunk_cffi import ffi, lib
from .body import Body
from .transform import Transform
from .vec2d import Vec2d
_DrawFlags = int
class SpaceDebugColor(NamedTuple):
    # ***<module>.SpaceDebugColor: Failure: Different bytecode
    """Color tuple used by the debug drawing API."""
    def as_int(self) -> tuple[int, int, int, int, int]:
        """Return the color as a tuple of ints, where each value is rounded.\n\n>>> SpaceDebugColor(0, 51.1, 101.9, 255).as_int()\n(0, 51, 102, 255)\n"""
        return (round(self[0]), round(self[1]), round(self[2]), round(self[3]))
    def as_float(self) -> tuple[float, float, float, float, float]:
        """Return the color as a tuple of floats, each value divided by 255.\n\n>>> SpaceDebugColor(0, 51, 102, 255).as_float()\n(0.0, 0.2, 0.4, 1.0)\n"""
        return (self[0] / 255.0, self[1] / 255.0, self[2] / 255.0, self[3] / 255.0)
    r: float
class SpaceDebugDrawOptions(object):
    # ***<module>.SpaceDebugDrawOptions: Failure: Different bytecode
    """SpaceDebugDrawOptions configures debug drawing.\n\nIf appropriate its usually easy to use the supplied draw implementations\ndirectly: pymunk.pygame_util, pymunk.pyglet_util and pymunk.matplotlib_util.\n"""
    DRAW_SHAPES = lib.CP_SPACE_DEBUG_DRAW_SHAPES
    pass
    DRAW_CONSTRAINTS = lib.CP_SPACE_DEBUG_DRAW_CONSTRAINTS
    pass
    DRAW_COLLISION_POINTS = lib.CP_SPACE_DEBUG_DRAW_COLLISION_POINTS
    pass
    shape_dynamic_color = SpaceDebugColor(52, 152, 219, 255)
    shape_static_color = SpaceDebugColor(149, 165, 166, 255)
    shape_kinematic_color = SpaceDebugColor(39, 174, 96, 255)
    shape_sleeping_color = SpaceDebugColor(114, 148, 168, 255)
    def __init__(self) -> None:
        _options = ffi.new('cpSpaceDebugDrawOptions *')
        self._options = _options
        self._options.transform = Transform.identity()
        self.shape_outline_color = SpaceDebugColor(44, 62, 80, 255)
        self.constraint_color = SpaceDebugColor(142, 68, 173, 255)
        self.collision_point_color = SpaceDebugColor(231, 76, 60, 255)
        self._use_chipmunk_debug_draw = True
        _options.drawCircle = lib.ext_cpSpaceDebugDrawCircleImpl
        _options.drawSegment = lib.ext_cpSpaceDebugDrawSegmentImpl
        _options.drawFatSegment = lib.ext_cpSpaceDebugDrawFatSegmentImpl
        _options.drawPolygon = lib.ext_cpSpaceDebugDrawPolygonImpl
        _options.drawDot = lib.ext_cpSpaceDebugDrawDotImpl
        _options.colorForShape = lib.ext_cpSpaceDebugDrawColorForShapeImpl
        self.flags = SpaceDebugDrawOptions.DRAW_SHAPES | SpaceDebugDrawOptions.DRAW_CONSTRAINTS | SpaceDebugDrawOptions.DRAW_COLLISION_POINTS
    @property
    def shape_outline_color(self) -> SpaceDebugColor:
        """The outline color of shapes.\n\nShould be a tuple of 4 ints between 0 and 255 (r,g,b,a).\n\nExample:\n\n>>> import pymunk\n>>> s = pymunk.Space()\n>>> c = pymunk.Circle(s.static_body, 10)\n>>> s.add(c)\n>>> options = pymunk.SpaceDebugDrawOptions()\n>>> s.debug_draw(options)\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 10.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=149.0, g=165.0, b=166.0, a=255.0))\n>>> options.shape_outline_color = (10,20,30,40)\n>>> s.debug_draw(options)\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 10.0, SpaceDebugColor(r=10.0, g=20.0, b=30.0, a=40.0), SpaceDebugColor(r=149.0, g=165.0, b=166.0, a=255.0))\n\n"""
        return self._c(self._options.shapeOutlineColor)
    @shape_outline_color.setter
    def shape_outline_color(self, c: SpaceDebugColor) -> None:
        self._options.shapeOutlineColor = c
    @property
    def constraint_color(self) -> SpaceDebugColor:
        """The color of constraints.\n\nShould be a tuple of 4 ints between 0 and 255 (r,g,b,a).\n\nExample:\n\n>>> import pymunk\n>>> s = pymunk.Space()\n>>> b = pymunk.Body(1, 10)\n>>> j = pymunk.PivotJoint(s.static_body, b, (0,0))\n>>> s.add(j)\n>>> options = pymunk.SpaceDebugDrawOptions()\n>>> s.debug_draw(options)\ndraw_dot (5.0, Vec2d(0.0, 0.0), SpaceDebugColor(r=142.0, g=68.0, b=173.0, a=255.0))\ndraw_dot (5.0, Vec2d(0.0, 0.0), SpaceDebugColor(r=142.0, g=68.0, b=173.0, a=255.0))\n>>> options.constraint_color = (10,20,30,40)\n>>> s.debug_draw(options)\ndraw_dot (5.0, Vec2d(0.0, 0.0), SpaceDebugColor(r=10.0, g=20.0, b=30.0, a=40.0))\ndraw_dot (5.0, Vec2d(0.0, 0.0), SpaceDebugColor(r=10.0, g=20.0, b=30.0, a=40.0))\n\n"""
        return self._c(self._options.constraintColor)
    @constraint_color.setter
    def constraint_color(self, c: SpaceDebugColor) -> None:
        self._options.constraintColor = c
    @property
    def collision_point_color(self) -> SpaceDebugColor:
        """The color of collisions.\n\nShould be a tuple of 4 ints between 0 and 255 (r,g,b,a).\n\nExample:\n\n>>> import pymunk\n>>> s = pymunk.Space()\n>>> b = pymunk.Body(1,10)\n>>> c1 = pymunk.Circle(b, 10)\n>>> c2 = pymunk.Circle(s.static_body, 10)\n>>> s.add(b, c1, c2)\n>>> s.step(1)\n>>> options = pymunk.SpaceDebugDrawOptions()\n>>> s.debug_draw(options)\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 10.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=52.0, g=152.0, b=219.0, a=255.0))\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 10.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=149.0, g=165.0, b=166.0, a=255.0))\ndraw_segment (Vec2d(8.0, 0.0), Vec2d(-8.0, 0.0), SpaceDebugColor(r=231.0, g=76.0, b=60.0, a=255.0))\n>>> options.collision_point_color = (10,20,30,40)\n>>> s.debug_draw(options)\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 10.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=52.0, g=152.0, b=219.0, a=255.0))\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 10.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=149.0, g=165.0, b=166.0, a=255.0))\ndraw_segment (Vec2d(8.0, 0.0), Vec2d(-8.0, 0.0), SpaceDebugColor(r=10.0, g=20.0, b=30.0, a=40.0))\n"""
        return self._c(self._options.collisionPointColor)
    @collision_point_color.setter
    def collision_point_color(self, c: SpaceDebugColor) -> None:
        self._options.collisionPointColor = c
    def __enter__(self) -> None:
        return
    def __exit__(self, type: Optional[type[BaseException]], value: Optional[BaseException], traceback: Optional['TracebackType']) -> None:
        return
    def _c(self, color: ffi.CData) -> SpaceDebugColor:
        return SpaceDebugColor(color.r, color.g, color.b, color.a)
    @property
    def flags(self) -> _DrawFlags:
        """Bit flags which of shapes, joints and collisions should be drawn.\n\nBy default all 3 flags are set, meaning shapes, joints and collisions\nwill be drawn.\n\nExample using the basic text only DebugDraw implementation (normally\nyou would the desired backend instead, such as\n`pygame_util.DrawOptions` or `pyglet_util.DrawOptions`):\n\n>>> import pymunk\n>>> s = pymunk.Space()\n>>> b = pymunk.Body()\n>>> c = pymunk.Circle(b, 10)\n>>> c.mass = 3\n>>> s.add(b, c)\n>>> s.add(pymunk.Circle(s.static_body, 3))\n>>> s.step(0.01)\n>>> options = pymunk.SpaceDebugDrawOptions()\n\n>>> # Only draw the shapes, nothing else:\n>>> options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES\n>>> s.debug_draw(options)\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 10.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=52.0, g=152.0, b=219.0, a=255.0))\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 3.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=149.0, g=165.0, b=166.0, a=255.0))\n\n>>> # Draw the shapes and collision points:\n>>> options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES\n>>> options.flags |= pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS\n>>> s.debug_draw(options)\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 10.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=52.0, g=152.0, b=219.0, a=255.0))\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 3.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=149.0, g=165.0, b=166.0, a=255.0))\ndraw_segment (Vec2d(1.0, 0.0), Vec2d(-8.0, 0.0), SpaceDebugColor(r=231.0, g=76.0, b=60.0, a=255.0))\n\n"""
        return self._options.flags
    @flags.setter
    def flags(self, f: _DrawFlags) -> None:
        self._options.flags = f
    @property
    def transform(self) -> Transform:
        """The transform is applied before drawing, e.g for scaling or\ntranslation.\n\nExample:\n\n>>> import pymunk\n>>> s = pymunk.Space()\n>>> c = pymunk.Circle(s.static_body, 10)\n>>> s.add(c)\n>>> options = pymunk.SpaceDebugDrawOptions()\n>>> s.debug_draw(options)\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 10.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=149.0, g=165.0, b=166.0, a=255.0))\n>>> options.transform = pymunk.Transform.scaling(2)\n>>> s.debug_draw(options)\ndraw_circle (Vec2d(0.0, 0.0), 0.0, 20.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=149.0, g=165.0, b=166.0, a=255.0))\n>>> options.transform = pymunk.Transform.translation(2,3)\n>>> s.debug_draw(options)\ndraw_circle (Vec2d(2.0, 3.0), 0.0, 10.0, SpaceDebugColor(r=44.0, g=62.0, b=80.0, a=255.0), SpaceDebugColor(r=149.0, g=165.0, b=166.0, a=255.0))\n\n.. Note::\n    Not all tranformations are supported by the debug drawing logic.\n    Uniform scaling and translation are supported, but not rotation,\n    linear stretching or shearing.\n"""
        t = self._options.transform
        return Transform(t.a, t.b, t.c, t.d, t.tx, t.ty)
    @transform.setter
    def transform(self, t: Transform) -> None:
        self._options.transform = t
    def draw_circle(self, pos: Vec2d, angle: float, radius: float, outline_color: SpaceDebugColor, fill_color: SpaceDebugColor) -> None:
        print('draw_circle', (pos, angle, radius, outline_color, fill_color))
    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        print('draw_segment', (a, b, color))
    def draw_fat_segment(self, a: Vec2d, b: Vec2d, radius: float, outline_color: SpaceDebugColor, fill_color: SpaceDebugColor) -> None:
        print('draw_fat_segment', (a, b, radius, outline_color, fill_color))
    def draw_polygon(self, verts: Sequence[Vec2d], radius: float, outline_color: SpaceDebugColor, fill_color: SpaceDebugColor) -> None:
        print('draw_polygon', (verts, radius, outline_color, fill_color))
    def draw_dot(self, size: float, pos: Vec2d, color: SpaceDebugColor) -> None:
        print('draw_dot', (size, pos, color))
    def draw_shape(self, shape: 'Shape') -> None:
        print('draw_shape', shape)
    def color_for_shape(self, shape: 'Shape') -> SpaceDebugColor:
        if hasattr(shape, 'color'):
            return SpaceDebugColor(*shape.color)
        else:
            color = self.shape_dynamic_color
            if shape.body != None:
                if shape.body.body_type == Body.STATIC:
                    color = self.shape_static_color
                    return color
                else:
                    if shape.body.body_type == Body.KINEMATIC:
                        color = self.shape_kinematic_color
                        return color
                    else:
                        if shape.body.is_sleeping:
                            color = self.shape_sleeping_color
            return color
    DRAW_SHAPES: ClassVar[_DrawFlags]