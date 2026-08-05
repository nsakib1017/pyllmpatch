from __future__ import annotations

# --- loads -------------------------------------------------------------------------------
LOCAL_LOAD_OPS = frozenset({
    "LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_AND_CLEAR", "LOAD_FAST_LOAD_FAST",
    "LOAD_FAST_BORROW", "LOAD_FAST_BORROW_LOAD_FAST_BORROW",  # 3.14+
})
CONST_LOAD_OPS = frozenset({
    "LOAD_CONST",
    "LOAD_SMALL_INT",  # 3.14+: small ints get a dedicated load op
})
COMMON_CONST_OPS = frozenset({"LOAD_COMMON_CONSTANT"})  # 3.14+ (AssertionError, tuple, all, ...)
NAME_LOAD_OPS = frozenset({"LOAD_NAME", "LOAD_GLOBAL"})
# LOAD_METHOD was folded into LOAD_ATTR (method bit) in 3.12+; both mean "load an attribute/method".
ATTR_LOAD_OPS = frozenset({"LOAD_ATTR", "LOAD_METHOD"})

# --- jumps -------------------------------------------------------------------------------
CONDITIONAL_JUMP_OPS = frozenset({
    "POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", "POP_JUMP_IF_NONE", "POP_JUMP_IF_NOT_NONE",
    "POP_JUMP_FORWARD_IF_FALSE", "POP_JUMP_FORWARD_IF_TRUE",
    "POP_JUMP_FORWARD_IF_NONE", "POP_JUMP_FORWARD_IF_NOT_NONE",
    "POP_JUMP_BACKWARD_IF_FALSE", "POP_JUMP_BACKWARD_IF_TRUE",
    "POP_JUMP_BACKWARD_IF_NONE", "POP_JUMP_BACKWARD_IF_NOT_NONE",
    "JUMP_IF_FALSE_OR_POP", "JUMP_IF_TRUE_OR_POP",
    "JUMP_IF_FALSE", "JUMP_IF_TRUE",  # 3.14+
})
UNCONDITIONAL_JUMP_OPS = frozenset({
    "JUMP", "JUMP_ABSOLUTE", "JUMP_FORWARD", "JUMP_BACKWARD",
    "JUMP_BACKWARD_NO_INTERRUPT",
    "JUMP_NO_INTERRUPT",  # 3.14+
})

# --- returns -----------------------------------------------------------------------------
# RETURN_CONST (3.12-3.13) was removed in 3.14 -> emitted as LOAD_CONST + RETURN_VALUE.
RETURN_OPS = frozenset({"RETURN_VALUE", "RETURN_CONST"})


def is_jump(opname: str) -> bool:
    return opname in CONDITIONAL_JUMP_OPS or opname in UNCONDITIONAL_JUMP_OPS




def is_local_load(opname: str) -> bool:
    return opname in LOCAL_LOAD_OPS

def is_const_load(opname: str) -> bool:
    return opname in CONST_LOAD_OPS

def is_attr_load(opname: str) -> bool:
    return opname in ATTR_LOAD_OPS

def is_conditional_jump(opname: str) -> bool:
    return opname in CONDITIONAL_JUMP_OPS

def is_unconditional_jump(opname: str) -> bool:
    return opname in UNCONDITIONAL_JUMP_OPS

def is_return(opname: str) -> bool:
    return opname in RETURN_OPS
