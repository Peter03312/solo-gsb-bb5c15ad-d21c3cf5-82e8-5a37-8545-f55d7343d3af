"""快照 Merkle 封存：跨语言可复算的 SHA-256 叶子摘要 / 树根 / 包含证明。

下游导出字幕时需要证明“单条内容确属某次发布快照”。本模块只对快照冻结
数据做纯函数计算，不读写任何可变状态，相同输入永远得到相同输出。

方案标识 ``snapshot-merkle:v1``，分三层定义，任何语言都可照此复算：

1. 规范序列化（canonical JSON 子集，按 UTF-8 编码为字节后参与哈希）
   - 无任何多余空白：成员分隔符为 ``,` / `:`（等价 JSON 紧凑模式）；
   - 仅支持 null / true / false / 整数 / 字符串 / 数组 / 对象：
     整数为十进制 ASCII（无指数、无正号、无前导零）；
   - 对象为固定键集、固定键序（见 ``leaf_entry``），不做键排序；
   - 字符串仅转义 `"`、``\\``、退格 / 换页 / 换行 / 回车 / 制表
     （``\\b \\f \\n \\r \\t``）以及 U+0000..U+001F（``\\u00xx``，
     四位小写十六进制）；其余所有码点（含中文、emoji、U+007F、
     U+2028 等）原样输出，不转义非 ASCII，不做任何 Unicode 归一化。
   该规则与 RFC 8785（JCS）的字符串转义一致；其他语言实现注意：
   Go 需 ``json.Encoder.SetEscapeHTML(false)``，JavaScript 的
   ``JSON.stringify`` 对 U+2028/U+2029 原样输出即符合本方案。

2. 域分离哈希（输出均为 32 字节摘要的小写十六进制）
   - 叶子：SHA-256(UTF-8("snapshot-merkle:v1:leaf:") || 规范序列化字节)
   - 内部：SHA-256(UTF-8("snapshot-merkle:v1:node:") ||
                   左子摘要原始 32 字节 || 右子摘要原始 32 字节)
     （内部节点拼接的是摘要的原始字节，不是十六进制字符串）
   - 空快照固定根：SHA-256(UTF-8("snapshot-merkle:v1:empty"))，
     常量见 ``EMPTY_ROOT``；所有 watermark=0 的快照共享该固定根。

3. 树结构
   - 叶子按 seq=1..watermark 顺序排列（字幕与缺口统一占一个叶位）；
   - 每层自左向右两两配对；某层节点数为奇数且大于 1 时，把该层最后
     一个节点复制一份与自身配对（H(x,x)）；
   - 恰好一个叶子的树根就是该叶子摘要本身（不再复制 / 再哈希）；
   - 包含证明按层给出兄弟摘要与相对方向：叶子在左（index 偶）时
     兄弟在右（side="right"，复制位 duplicated=true 表示该兄弟是
     末叶复制，摘要与当前节点相同）；叶子在右（index 奇）时兄弟在
     左（side="left"，永不复制）。
"""

import hashlib

#: 方案版本标识，随响应返回，供下游区分算法版本
SCHEME = "snapshot-merkle:v1"

#: 叶子域前缀（UTF-8 字节，直接与规范序列化负载拼接）
LEAF_PREFIX = b"snapshot-merkle:v1:leaf:"

#: 内部节点域前缀（UTF-8 字节，后接左右各 32 字节原始摘要）
NODE_PREFIX = b"snapshot-merkle:v1:node:"

#: 空快照固定根：所有 watermark=0 的快照均返回该值
EMPTY_ROOT = hashlib.sha256(b"snapshot-merkle:v1:empty").hexdigest()

#: 摘要算法与编码（随响应返回，便于离线工具断言）
HASH_ALGORITHM = "SHA-256"
HASH_ENCODING = "hex"

#: 叶子负载的固定键序（字幕 / 缺口共用同一 9 键集，缺失值用 null）
LEAF_KEYS = (
    "snapshot_id",
    "seq",
    "kind",
    "revision",
    "original_text",
    "effective_text",
    "correction_version",
    "gap_start",
    "gap_end",
)

_SHORT_ESCAPES = {
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _encode_string(s: str) -> str:
    """按规范序列化规则转义 JSON 字符串（含两侧引号，尚未 UTF-8 编码）。"""
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch in _SHORT_ESCAPES:
            out.append(_SHORT_ESCAPES[ch])
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            # U+007F 及以上码点原样输出，不做 HTML / 非 ASCII 转义
            out.append(ch)
    out.append('"')
    return "".join(out)


def canonical_dumps(obj) -> bytes:
    """把叶子负载（dict / list / str / int / None / bool）序列化为规范字节。

    本方案只使用上述 JSON 子集；遇到不支持的类型（浮点 / bytes 等）直接
    抛 TypeError——叶子负载中所有数字都是整数。
    """
    if obj is None:
        return b"null"
    if obj is True:
        return b"true"
    if obj is False:
        return b"false"
    if isinstance(obj, int):
        return str(obj).encode("ascii")
    if isinstance(obj, str):
        return _encode_string(obj).encode("utf-8")
    if isinstance(obj, (list, tuple)):
        return b"[" + b",".join(canonical_dumps(item) for item in obj) + b"]"
    if isinstance(obj, dict):
        pairs = []
        for key, value in obj.items():
            pairs.append(
                _encode_string(key).encode("utf-8")
                + b":"
                + canonical_dumps(value)
            )
        return b"{" + b",".join(pairs) + b"}"
    raise TypeError(f"规范序列化不支持类型 {type(obj)!r}")


def leaf_entry(snapshot_id: str, row) -> dict:
    """从 snapshot_items 行构造固定 9 键的叶子负载（键序固定）。

    字幕行：revision / original_text / effective_text / correction_version
    为冻结值，缺口端点为 null；缺口行：四个字幕字段为 null（注意是 JSON
    null，不是空字符串），gap_start / gap_end 为冻结的裁决区间端点。
    """
    if row["kind"] == "gap":
        return {
            "snapshot_id": snapshot_id,
            "seq": row["seq"],
            "kind": "gap",
            "revision": None,
            "original_text": None,
            "effective_text": None,
            "correction_version": None,
            "gap_start": row["start_seq"],
            "gap_end": row["end_seq"],
        }
    return {
        "snapshot_id": snapshot_id,
        "seq": row["seq"],
        "kind": "subtitle",
        "revision": row["revision"],
        "original_text": row["original_text"],
        "effective_text": row["effective_text"],
        "correction_version": row["correction_version"],
        "gap_start": None,
        "gap_end": None,
    }


def hash_leaf(entry: dict) -> str:
    """叶子摘要：SHA-256(LEAF_PREFIX || 规范序列化字节)，小写十六进制。"""
    return hashlib.sha256(LEAF_PREFIX + canonical_dumps(entry)).hexdigest()


def hash_node(left_hex: str, right_hex: str) -> str:
    """内部节点摘要：域前缀后拼接左右摘要的**原始 32 字节**。"""
    preimage = NODE_PREFIX + bytes.fromhex(left_hex) + bytes.fromhex(right_hex)
    return hashlib.sha256(preimage).hexdigest()


def build_levels(leaf_hashes: list) -> list:
    """自底向上构造各层节点摘要；空列表返回 []，单叶返回单层。"""
    if not leaf_hashes:
        return []
    levels = [list(leaf_hashes)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        nxt = []
        for i in range(0, len(current), 2):
            left = current[i]
            # 奇数层（长度 > 1）复制末叶与自身配对
            right = current[i + 1] if i + 1 < len(current) else left
            nxt.append(hash_node(left, right))
        levels.append(nxt)
    return levels


def merkle_root(leaf_hashes: list) -> str:
    """计算树根；空叶子列表返回空快照固定根。"""
    if not leaf_hashes:
        return EMPTY_ROOT
    return build_levels(leaf_hashes)[-1][0]


def inclusion_proof(leaf_hashes: list, index: int) -> list:
    """生成 index（0 基，等于 seq-1）处叶子的兄弟路径（自叶向根）。

    每步含：level（层号，叶子层为 0）、side（兄弟相对当前节点的方向）、
    hash（兄弟摘要）、duplicated（该兄弟是否为奇数层末叶的复制）。
    """
    levels = build_levels(leaf_hashes)
    path = []
    idx = index
    for level, nodes in enumerate(levels[:-1]):
        if idx % 2 == 0:
            sibling_index = idx + 1
            duplicated = sibling_index >= len(nodes)
            sibling = nodes[sibling_index] if not duplicated else nodes[idx]
            side = "right"
        else:
            duplicated = False
            sibling = nodes[idx - 1]
            side = "left"
        path.append(
            {
                "level": level,
                "side": side,
                "hash": sibling,
                "duplicated": duplicated,
            }
        )
        idx //= 2
    return path


#: SHA-256 摘要的十六进制长度（32 字节）
_DIGEST_HEX_LEN = 64

_HEX_DIGITS = frozenset("0123456789abcdef")

#: 包含证明每一步必须且仅能包含的键（缺键 / 多键均视为结构错误）
_PROOF_STEP_KEYS = frozenset(("level", "side", "hash", "duplicated"))


def _is_plain_int(value) -> bool:
    """判定是否为非布尔整数（bool 虽是 int 子类，但语义上不是序号 / 计数）。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256_hex(value) -> bool:
    """严格判定是否为本方案规定的 SHA-256 摘要：恰好 64 位小写十六进制。

    不使用 ``bytes.fromhex`` 判定——它接受空白分隔与大写，且对非法字符
    / 奇数长度直接抛 ValueError；验签入口对格式错误只能返回 False。
    """
    return (
        isinstance(value, str)
        and len(value) == _DIGEST_HEX_LEN
        and all(ch in _HEX_DIGITS for ch in value)
    )


def verify_inclusion_proof(
    *,
    leaf_hash: str,
    index: int,
    leaf_count: int,
    path: list,
    root_hex: str,
) -> bool:
    """严格复算包含证明：方向与复制位均由 (index, leaf_count) 重新推导。

    供离线验真参考实现——不信任路径自报的 side / duplicated，只信兄弟
    摘要；任何一步不一致或最终摘要不等于树根都返回 False。单叶树
    （leaf_count=1）的路径为空，树根应恰为叶子摘要。

    对任何格式 / 结构不合法的输入（摘要不是 64 位小写十六进制、路径
    步骤不是对象或缺 / 多键、路径缺步或多步、序号 / 叶总数类型错误等）
    一律返回 False，绝不向调用方泄漏 ``bytes.fromhex`` 等解析异常——
    调用方只需按布尔结论区分有效 / 无效证明。
    """
    # 入参结构与格式校验：不合法即按无效证明处理，避免后续解析抛异常
    if not _is_plain_int(leaf_count) or not _is_plain_int(index):
        return False
    if leaf_count < 1 or not 0 <= index < leaf_count:
        return False
    if not isinstance(path, (list, tuple)):
        return False
    if not _is_sha256_hex(leaf_hash) or not _is_sha256_hex(root_hex):
        return False

    level_len = leaf_count
    idx = index
    current = leaf_hash
    for expected_level, step in enumerate(path):
        if level_len == 1:
            # 多步：已经到达唯一的根节点，路径不应再给出兄弟
            return False
        if not isinstance(step, dict):
            # 路径步骤必须是对象（非对象步骤无法安全取键，直接判无效）
            return False
        if set(step) != _PROOF_STEP_KEYS:
            # 缺键 / 多键均属结构错误
            return False
        level = step["level"]
        side = step["side"]
        sibling = step["hash"]
        duplicated = step["duplicated"]
        if not _is_plain_int(level) or level < 0 or level != expected_level:
            return False
        if not isinstance(side, str):
            return False
        if not _is_sha256_hex(sibling):
            # 非法十六进制兄弟摘要 -> 验证失败，而非抛 ValueError
            return False
        if not isinstance(duplicated, bool):
            return False
        # 复制只可能发生在：index 为偶数、且它就是本层最后一个节点
        expect_duplicated = idx % 2 == 0 and idx + 1 >= level_len
        if duplicated != expect_duplicated:
            return False
        if idx % 2 == 0:
            if side != "right":
                return False
            if duplicated:
                if sibling != current:
                    return False
                left, right = current, current
            else:
                left, right = current, sibling
        else:
            if side != "left":
                return False
            left, right = sibling, current
        # 三个摘要均已通过 _is_sha256_hex 校验，fromhex 不会再抛异常
        current = hash_node(left, right)
        level_len = (level_len + 1) // 2
        idx //= 2
    # 缺步（循环结束仍未折到唯一根）/ idx 未归 0 / 根不符都在此判失败
    return level_len == 1 and idx == 0 and current == root_hex
