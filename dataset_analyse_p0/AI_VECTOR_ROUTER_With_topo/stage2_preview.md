Phase 2 拓扑标注 Preview Tool 实现方案
1. 文件信息
● 输出文件：dataset_analyse_p0/AI_VECTOR_ROUTER_With_topo/action_stage2_preview_tool.py
● 数据来源：annotations_topo/{字体名}_topo.json（每个字符一个 bundle）
2. 数据结构说明
每个 char_bundle 含：
● glyph_info: hex_key, char
● strokes: bezier_id, mother_bezier(4,2 控制点), width_bezier(4,), stroke_type, length, bbox
● topology_events: {type:E2E/T/X, stroke_a/b, t_a/b, guest/host, angle, position}
● cycles: {cycle_id, members, orientation}
● edit_history: {action:SNAP/T_ATTACH/CONTROL_MOVE, stroke/guest, endpoint, host_stroke...}
3. 时光机帧重建逻辑
Phase 2 的 edit_history 只记录"增量操作"，不像 Phase 1 每步都存全量 edges 快照。需要：
1. 用 strokes（最终状态）作为基准
2. 将 edit_history 逆向 replay 还原每步之前的状态：
	○ CONTROL_MOVE: 将 after 坐标改回 before 坐标（逆向）
	○ SNAP: 将 after 改回 before（逆向）
	○ T_ATTACH: 将 after 改回 before（逆向）
3. 通过从最终状态逆向推导得到每一帧
或者更简单：从初始状态正向 replay：
● 初始状态 = 对每一步 edit_history 执行反向操作得到 t=0 时的 strokes
● 然后逐步正向应用每个 edit_history 操作
4. 实现架构
4.1 辅助函数
def bezier_at_t(P, t):
    """三次贝塞尔采样"""

def compute_topo_from_strokes(strokes, canvas_size=400):
    """
    从 strokes 列表计算拓扑关系（E2E/T/X/cycles）
    仿照 topo_editor_workspace.py 的 update_topology_text 逻辑
    返回 HTML 字符串
    """

def get_hex_color(bezier_id):
    """用 tab20 cmap 给笔画着色"""

def replay_strokes(final_strokes, edit_history):
    """
    将 edit_history 逆向应用得到初始状态，然后正向 replay，
    返回 [(strokes_snapshot, action_name)] 列表
    """
4.2 StateRenderer（仿照 Phase 1 preview）
● render_stroke_image(strokes, size): 用 mother_bezier 画彩色贝塞尔曲线，每条笔画不同颜色
4.3 主界面结构（完全复用 action_preview_tool.py 的框架）
● PreviewAppMain: 主窗口，左侧字体列表（已标注/未标注）
● Phase2PreviewWorkspace: 右侧工作区
	○ load_gallery(): 读取 annotations_topo 下所有 *_topo.json，按字体分组，展示字符缩略图
	○ open_timeline(hex_key): 打开某字符的时光机
		■ 区域 A: 固定 Original 参考图（初始状态 strokes）
		■ 区域 B: 横向滚动帧列表，每帧 = 彩色笔画图 + 拓扑文本
4.4 拓扑文本格式（仿照 topo_editor_workspace.py 的 update_topology_text）
<b>📊 实时拓扑状态反馈</b>
[端点对接]：<span color=笔画色>1</span>-<span color=笔画色>2</span>
[T型搭接]：<span>2</span> 搭在 <span>1</span> 上
[X型交叉]：<span>3</span> 交叉 <span>4</span>
[闭环结构]：<span>1</span> <span>2</span> 属同一环
5. edit_history 的 replay 机制
edit_history 每条记录都保存了 before/after 坐标，可以精确重建每一步：
初始 strokes = copy(final_strokes)
# 逆序还原到初始状态
for op in reversed(edit_history):
    apply_reverse(op, strokes)

frames = [(copy(strokes), "Initial")]
# 正向 replay
for op in edit_history:
    apply_forward(op, strokes)
    frames.append((copy(strokes), op["action"]))
每个 op 的操作：
● CONTROL_MOVE: strokes[stroke-1]["mother_bezier"][P_idx] 改为 after/before
● SNAP: 同上，after/before 是端点坐标
● T_ATTACH: 同上
6. 关键细节
● 字体列表读取：扫描 annotations_topo/ 目录下的所有 *_topo.json
● 缩略图：使用 render_unicode_glyph 渲染原始字形（需要匹配字体文件路径）
● CANVAS_SIZE=400 坐标系与字体渲染一致

