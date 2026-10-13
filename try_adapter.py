from tryon.avatar_representation import build_avatar_representation

rep = build_avatar_representation()
print(rep.left_underarm, rep.right_underarm)
r = rep.underarm_result
print(r.method.value, r.revision, r.left.source.value, r.right.source.value)