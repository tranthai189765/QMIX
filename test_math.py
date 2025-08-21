import math

def normalize_angle(angle):
    """Đưa góc về [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle

def compute_alpha(agent_pos, agent_dir, target_pos):
    """
    Tính alpha: góc giữa hướng agent và target.
    agent_pos: (x, y)
    agent_dir: góc rad, hướng nhìn của agent
    target_pos: (x, y)
    """
    dx, dy = target_pos[0] - agent_pos[0], target_pos[1] - agent_pos[1]
    angle_to_target = math.atan2(dy, dx)
    alpha = normalize_angle(angle_to_target - agent_dir)
    return alpha

def compute_theta(agent_pos, seen_targets):
    """
    Tính theta: góc bé nhất bao tất cả seen targets.
    agent_pos: (x, y)
    seen_targets: list[(x, y)]
    """
    if not seen_targets:
        return 0.0

    # Tính góc tuyệt đối đến từng target
    angles = [math.atan2(t[1] - agent_pos[1], t[0] - agent_pos[0]) for t in seen_targets]
    angles.sort()

    # Tính khoảng cách góc liên tiếp
    diffs = []
    for i in range(len(angles)):
        j = (i + 1) % len(angles)
        diff = normalize_angle(angles[j] - angles[i])
        if diff < 0:
            diff += 2 * math.pi
        diffs.append(diff)

    # Khoảng lớn nhất là vùng KHÔNG có target
    max_gap = max(diffs)
    theta = 2 * math.pi - max_gap
    return theta

# ---------------- Ví dụ ----------------
if __name__ == "__main__":
    agent_pos = (0, 0)
    agent_dir = math.radians(0)  # hướng sang phải (trục x+)

    seen_targets = [(0, 0), (-2, 0), (0, -3), (0, 3)]
    unseen_target = (0, 2)

    # Tính alpha cho từng target
    print("Alpha (seen targets):")
    for t in seen_targets:
        print(f"Target {t}: alpha = {math.degrees(compute_alpha(agent_pos, agent_dir, t)):.2f}°")

    print(f"\nTheta (bao toàn bộ seen targets) = {math.degrees(compute_theta(agent_pos, seen_targets)):.2f}°")

    # Kiểm tra target chưa thấy
    alpha_unseen = compute_alpha(agent_pos, agent_dir, unseen_target)
    print(f"\nUnseen target {unseen_target}: alpha = {math.degrees(alpha_unseen):.2f}°")
