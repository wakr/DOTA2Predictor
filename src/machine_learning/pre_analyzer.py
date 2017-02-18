"""
    helpers
"""


def get_selected_heroes(picks):
    h = []
    for ID, b in enumerate(picks):
        if b:
            h.append(ID+1)
    return h


"""
    combiners - creates crossfeature lists
    [d1,d2,d3,d4,d5, r1,r2,r3,r4,r5]
"""


def get_winlosses(heroes, wl_data):
    crossF = []
    for ID in heroes:
        crossF.append(wl_data[ID])
    return crossF


def get_synergy_rate(heroes, synergypairs):
    crossF = [0, 0]  # dire, radiant
    for pair in synergypairs:
        if pair[0] in heroes[:5] and pair[1] in heroes[:5]:
            crossF[0] += 1
        if pair[0] in heroes[5:] and pair[1] in heroes[5:]:
            crossF[1] += 1
    return crossF


def get_distr_amount(heroes, top10):
    crossF = [0, 0]  # dire, radiant
    top10IDs = [h[0] for h in top10]
    for dheroID in heroes[:5]:
        if dheroID in top10IDs:
            crossF[0] += 1
    for rheroID in heroes[5:]:
        if rheroID in top10IDs:
            crossF[1] += 1
    return crossF


def get_heroSynergy_Diff(heroes, synergy_matrix):
    dSum = 0
    rSum = 0
    for dheroID in heroes[:5]:
        for dpairID in heroes[:5]:
            dSum += synergy_matrix[dheroID - 1, dpairID - 1]
    for rheroID in heroes[5:]:
        for rpairID in heroes[5:]:
            rSum += synergy_matrix[rheroID - 1, rpairID - 1]
    return [dSum - rSum]

def get_counterSynergy(heroes, counter_matrix):
    dSum = 0
    for dheroID in heroes[:5]:
        for rheroID in heroes[5:]:
            dSum += counter_matrix[dheroID - 1, rheroID - 1]
    return [dSum]
