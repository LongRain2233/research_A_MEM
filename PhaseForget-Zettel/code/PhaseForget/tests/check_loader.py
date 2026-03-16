from phaseforget.evaluation.loaders.locomo import LoCoMoLoader
loader = LoCoMoLoader()
sessions = loader.load("dataset/locomo10.json")
print("Total benchmark sessions:", len(sessions), " (should be 10, one per record)")
s = sessions[0]
print("session_id:", s["session_id"])
print("Total dialogue turns (all sessions merged):", len(s["dialogue"]))
print("QA questions:", len(s["questions"]))
print("Original session_N count:", s["meta"]["num_sessions"])
