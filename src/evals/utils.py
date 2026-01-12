from azul_engine.serialization import state_from_dict


def state_from_snapshot(snapshot: dict):
    return state_from_dict(snapshot)
