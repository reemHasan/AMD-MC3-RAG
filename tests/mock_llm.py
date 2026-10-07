"""Scripted stand-in for the VLM: behaves like a *competent* model on the sample questions.
It validates the plumbing (parsing, retrieval, multi-hop, guards, IO), NOT model quality."""
import json, os, re

VISION = {
    "backplane_pinout.png": ("Backplane connector J1\nB12 = PWR_GOOD\nB14 = THERM_ALERT#\nB15 = FAN_TACH", "THERM_ALERT", "B14"),
    "asset_label.jpg": ("Orrery Systems\nMODEL TQ-40\nBOARD REVISION REV-C2\nSN OS4-118823-77", "revision", "REV-C2"),
}

class MockLLM:
    def generate(self, prompt, image=None, max_new_tokens=128, tag=""):
        if image is not None:
            name = os.path.basename(tag)
            if name not in VISION: return ""
            text, key, val = VISION[name]
            if prompt.startswith("Transcribe"): return text
            q = prompt.split("Question:")[1]
            return text + (f"\nANSWER: {val}" if key.lower() in q.lower() else "\nANSWER: NONE")
        if prompt.startswith("Question:"): return ""            # keyword expansion
        q = re.search(r"QUESTION: (.*)", prompt).group(1)
        ex = {}
        for m in re.finditer(r"\[E(\d+)\] (\S+) \| ([^\n]*)\n(.*?)(?=\n\n\[E\d+\]|\Z)", prompt, re.S):
            ex[int(m.group(1))] = (m.group(2), m.group(3), m.group(4))
        def find(pat, skip_withdrawn=True):
            for n, (f, loc, body) in ex.items():
                if skip_withdrawn and "WITHDRAWN" in loc: continue
                m = re.search(pat, body)
                if m: return n, m.group(1)
        def ans(hit, extra=()):
            if not hit: return json.dumps({"status": "none", "sources": []})
            return json.dumps({"status": "answer", "answer": hit[1], "sources": [hit[0], *extra]})
        ql = q.lower()
        if "junction temperature" in ql: return ans(find(r"junction temperature \(Tj max\): (\d+)"))
        if "customer sampling" in ql: return ans(find(r"TQ-60 \| Customer sampling \| (Q\d FY\d+)"))
        if "fan assembly" in ql: return ans(find(r"(?i)part number: (ORR-FAN-\d+-\w)[^\n]*TQ-40"))
        if "ticket orr-1847" in ql: return ans(find(r"ORR-1847[^\n]*fixed_in: (\S+)"))
        if "error code" in ql: return ans(find(r"(E\d{4}) THERMAL"))
        if "batch timeout" in ql: return ans(find(r"BATCH_TIMEOUT_S = (\d+)"))
        if "therm_alert" in ql or "board revision" in ql: return ans(find(r"ANSWER: (\S+)"))
        if "firmware release fixed" in ql:
            hit = find(r"ORR-1847[^\n]*fixed_in: (\S+)")
            if hit: 
                log = find(r"known issue (ORR-\d+)")
                return ans(hit, [log[0]] if (log and not os.environ.get('MOCK_FORGET')) else [])          # a sloppy model may forget this
            log = find(r"known issue (ORR-\d+)")
            if log: return json.dumps({"status": "search", "search": log[1], "sources": [log[0]]})
        if "unit price" in ql and os.environ.get("MOCK_HALLUCINATE"):
            return json.dumps({"status": "answer", "answer": "6412", "sources": [1]})
        return json.dumps({"status": "none", "sources": []})
