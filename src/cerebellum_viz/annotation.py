"""
Blinded annotation campaign : protocol, project (master, rater keys, hashes) and one rater's session.
Nothing here needs napari ; the form lives in cerebellum_viz.viewer.
"""

from datetime import datetime
from pathlib import Path
import copy
import hashlib
import json
import os
import re
import shutil

import numpy as np
import pandas as pd

from .data import DEFAULT_LAYOUT, Dataset, Subject

PROTOCOLS_DIR = Path(__file__).parent / "protocols"


def protocol_file(version : int) -> Path :
    """Path of a protocol shipped with the package (protocols/protocol_vN.json)."""
    path = PROTOCOLS_DIR / f"protocol_v{version}.json"
    if not path.exists() :
        shipped = sorted(p.name for p in PROTOCOLS_DIR.glob("protocol_v*.json"))
        raise FileNotFoundError(f"No shipped protocol v{version}. Available : {shipped}")
    return path


class Protocol :
    REQUIRED_KEYS = {"version", "items", "extra_choices", "comment_required_for", "quality"}

    def __init__(self, spec: dict) :
        # Copy so that later changes to the caller's dict cannot bypass validation
        self.spec = copy.deepcopy(spec)
        missing = self.REQUIRED_KEYS - self.spec.keys()
        if missing :
            raise KeyError(f"Protocol is missing required keys : {sorted(missing)}")
        self.version = self.spec["version"]
        self.validate()

    @classmethod
    def from_file(cls, path : str | Path) :
        path = Path(path)
        with open(path, "r") as f :
            spec = json.load(f)
        return cls(spec)

    def valid_codes(self, item) -> dict:
        # Item-specific choices + extra choices (ND, S) shared by all items
        return {**self.spec["items"][item]["choices"], **self.spec["extra_choices"]}

    def label(self, item, code):
        return self.valid_codes(item)[str(code)]

    def check_answer(self, item, code, quality, comment) :
        # JSON keys are always strings
        code = str(code)
        if item not in self.spec["items"] :
            raise ValueError(f"Item don't exist in protocol. \nAvailable items in this protocol : \n{list(self.spec['items'])} ")
        valid = self.valid_codes(item)
        if code not in valid :
            raise ValueError(f"Code '{code}' don't exist in this item ({item}). \nAvailable codes in this item : \n{list(valid)} ")
        if code in self.spec["comment_required_for"] and not comment :
            raise ValueError(f"Comment is empty for a code with required comment ({item} : {code}).")
        if quality not in self.spec["quality"] :
            raise ValueError(f"{quality} is not a quality tag.")

    def validate(self):
        if not isinstance(self.version, int) :
            raise TypeError("Version should an integer.")

        extras = set(self.spec["extra_choices"])
        for item in self.spec["items"].keys() :
            choices = self.spec["items"][item]["choices"]
            # Collision between item codes and extra codes would make answers ambiguous
            clash = extras & set(choices)
            if clash :
                raise ValueError(f"Item '{item}' has codes colliding with extra choices : {sorted(clash)}")
            # n_choices : min = 2
            n_choices = len(choices)
            if n_choices < 2 :
                raise ValueError(f"Item '{item}' don't have enough choices : {n_choices}")

        # comments are required for extra codes only
        for code in self.spec["comment_required_for"] :
            if code not in extras :
                raise ValueError(f"Comments required for a code that doesn't exist in extra choices : {code}")

        if not self.spec["quality"] :
            raise ValueError("Quality rating comments are empty...")

    def to_file(self, path : str | Path): 
        path = Path(path)
        if path.stem != f"protocol_v{self.version}":
            raise ValueError("File name doesn't match protocol formatting")
        with open(path, "x") as f :
            json.dump(self.spec, f, indent=2)


ANNOTATION_VERSION = 1

class AnnotationProject :
    def __init__(self, root : str | Path, meta : dict) :
        self.root = Path(root)
        # Meta is a dict (already existing : load, or to create : create)
        self.meta = meta


    @classmethod
    def create(cls, root : str | Path, subjects : list[str], dataset_root : str | Path, protocol_path : str | Path,
               n_repeats : int = 2, n_to_repeat : int = 30, min_gap : int = 50, seed : int = 42):
        root = Path(root)
        if (root / "project.json").exists() :
            raise FileExistsError("This project already olds a project.json file")

        # ---- All checks first : nothing is written before they pass ----
        if not Path(dataset_root).exists() :
            raise FileNotFoundError(f"Dataset root don't exist : {dataset_root}")
        subjects = [str(s) for s in subjects]
        if len(set(subjects)) != len(subjects) :
            dup = sorted({s for s in subjects if subjects.count(s) > 1})
            raise ValueError(f"Duplicated subjects in the list ({len(dup)}) : each subject must get a single pid")
        if not 1 <= n_to_repeat <= len(subjects) :
            raise ValueError(f"n_to_repeat must be in [1, {len(subjects)}], got {n_to_repeat}")
        if n_repeats < 2 :
            raise ValueError(f"n_repeats is the total number of appearances, must be >= 2, got {n_repeats}")
        if min_gap < 1 :
            raise ValueError(f"min_gap must be >= 1, got {min_gap}")
        # The appearances of a repeated pid span at least (n_repeats - 1) * min_gap positions
        n_cases = len(subjects) + n_to_repeat * (n_repeats - 1)
        if (n_repeats - 1) * min_gap > n_cases - 1 :
            raise ValueError(f"min_gap={min_gap} cannot be met : {n_repeats} appearances need "
                             f"{(n_repeats - 1) * min_gap + 1} cases, the order only has {n_cases}")
        protocol = Protocol.from_file(protocol_path) # Raises error if something is wrong

        # Doesn't exist, creating new meta
        meta = dict()
        meta["version"] = ANNOTATION_VERSION
        meta["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta["n_subjects"] = len(subjects)
        meta["n_cases"] = n_cases
        meta["n_to_repeat"] = int(n_to_repeat)
        meta["n_repeats"] = int(n_repeats)
        meta["min_gap"] = int(min_gap)
        meta["seed"] = int(seed)
        meta["dataset_root"] = str(dataset_root)

        rng = np.random.default_rng(seed)
        shuf_sub = rng.permutation(np.asarray(subjects))

        width = max(3, len(str(meta["n_subjects"])))
        p_ind = [f"p{str(i).zfill(width)}" for i in range(1, meta["n_subjects"] + 1)]
        meta["repeats"] = rng.choice(p_ind, n_to_repeat, replace=False).tolist()

        # ---- Writing : master, protocol copy, project.json last ----
        root.mkdir(parents=True, exist_ok=True)
        master_df = pd.DataFrame({"sid" : shuf_sub, "pid" : p_ind})
        master_df.to_csv(root / "master.csv", index=False)
        meta["master"] = {"file" : "master.csv", "sha256" : cls._hash(root / "master.csv")}

        rel_protocol = f"protocols/protocol_v{protocol.version}.json"
        (root / "protocols").mkdir(exist_ok=True)
        shutil.copyfile(protocol_path, root / rel_protocol)
        meta["protocol"] = {
            "current" : protocol.version,
            "files" : {str(protocol.version) : {
                "file" : rel_protocol,
                "sha256" : cls._hash(root / rel_protocol),
                "registered" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }},
        }
        meta["raters"] = {}

        # project.json is written last : its existence means the project is complete
        project = cls(root, meta)
        project._write_json()
        return project

    @classmethod
    def load(cls, root : str | Path):
        root = Path(root)
        if not (root / "project.json").exists() :
            raise FileNotFoundError("Project don't exist yet, use the AnnotationProject.create method")
        # File exist
        with open(root / "project.json", "r") as f :
            meta = json.load(f)

        if meta.get("version") != ANNOTATION_VERSION :
            raise ValueError(f"Project format version {meta.get('version')} does not match "
                             f"the code's ANNOTATION_VERSION ({ANNOTATION_VERSION})")

        # Checking hashes (all paths are relative to root)
        if not cls._check_hash(root / meta["master"]["file"], meta["master"]["sha256"]) :
            raise ValueError("Hash of the mapping file (pid/sid) doesn't match the one stored in the project !")

        for protocol_num, protocol_dict in meta["protocol"]["files"].items():
            if not cls._check_hash(root / protocol_dict["file"], protocol_dict["sha256"]) :
                raise ValueError(f"Hashes don't match for protocol : {protocol_num}")

        for rater, r_dict in meta["raters"].items() :
            if not cls._check_hash(root / r_dict["key"], r_dict["sha256"]) :
                raise ValueError(f"Hashes don't match for rater key : {rater}")

        project = cls(root, meta)
        project.protocol # Checks that current protocol is registered and valid
        return project

    def add_rater(self, name : str, seed : int | None = None):
        # ---- All checks first : nothing is written before they pass ----
        raters = self.meta["raters"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) :
            raise ValueError(f"Rater name '{name}' must only hold letters, digits, '_' or '-' (used as a file name)")
        if name in raters :
            raise ValueError(f"Cannot add this rater ({name}), the rater already exists.")
        if seed is None :
            seed = int(np.random.default_rng().integers(2**31))
        seed = int(seed)
        # Same seed = same order : the per-rater order would no longer be per rater
        used = {r : d["seed"] for r, d in raters.items() if d["seed"] == seed}
        if used :
            raise ValueError(f"Seed {seed} already used by rater(s) {list(used)}")
        rel_key = f"keys/{name}.csv"
        if (self.root / rel_key).exists() :
            raise FileExistsError(f"A key already exists for an unregistered rater : {self.root / rel_key}")

        order = self._build_order(self._pids(), self.meta["repeats"], self.meta["n_repeats"],
                                  self.meta["min_gap"], seed)

        # ---- Writing : key (never overwritten), then project.json ----
        (self.root / "keys").mkdir(exist_ok=True)
        key_df = pd.DataFrame({"case" : np.arange(1, len(order) + 1), "pid" : order})
        with open(self.root / rel_key, "x") as f :
            key_df.to_csv(f, index=False)
        raters[name] = {
            "key" : rel_key,
            "sha256" : self._hash(self.root / rel_key),
            "seed" : seed,
            "answers" : f"answers/{name}.csv",
            "created" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._write_json()
        return raters[name]

    def session(self, name : str, layout : dict = DEFAULT_LAYOUT) :
        return AnnotationSession(self, name, layout)

    @property
    def protocol(self):
        return self.protocol_for(self.meta["protocol"]["current"])

    def protocol_for(self, version) :
        files = self.meta["protocol"]["files"]
        if str(version) not in files :
            raise KeyError(f"Protocol version {version} not registered in this project : {list(files)}")
        return Protocol.from_file(self.root / files[str(version)]["file"])

    def update_protocol(self, protocol_path : str | Path, note : str | None = None) :
        # ---- All checks first ----
        protocol = Protocol.from_file(protocol_path) # Raises error if something is wrong
        current = self.meta["protocol"]["current"]
        if protocol.version <= current :
            raise ValueError(f"New protocol version ({protocol.version}) must be higher than the current one ({current})")
        rel_protocol = f"protocols/protocol_v{protocol.version}.json"
        if (self.root / rel_protocol).exists() :
            raise FileExistsError(f"A protocol file already exists for version {protocol.version} : {self.root / rel_protocol}")

        # ---- Writing : protocol copy, then project.json ----
        shutil.copyfile(protocol_path, self.root / rel_protocol)
        self.meta["protocol"]["files"][str(protocol.version)] = {
            "file" : rel_protocol,
            "sha256" : self._hash(self.root / rel_protocol),
            "registered" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note" : note,
        }
        self.meta["protocol"]["current"] = protocol.version
        self._write_json()

    def status(self):
        m = self.meta
        lines = [f"Project : {self.root}",
                 f"  {m['n_subjects']} subjects, {m['n_to_repeat']} repeated x{m['n_repeats']} "
                 f"(min gap {m['min_gap']}) -> {m['n_cases']} cases per rater",
                 f"  protocol : v{m['protocol']['current']} (registered : {sorted(m['protocol']['files'], key=int)})",
                 f"  raters ({len(m['raters'])}) :"]
        for name, r in m["raters"].items() :
            answered = (self.root / r["answers"]).exists()
            lines.append(f"    {name} : seed {r['seed']}, created {r['created']}, "
                         f"answers {'started' if answered else 'not started'}")
        print("\n".join(lines))

    @staticmethod
    def _build_order(pids : list[str], repeats : list[str], n_repeats : int, min_gap : int,
                     seed : int, max_tries : int = 100) -> list[str]:
        """
        Presentation order of one rater : every pid once, plus n_repeats - 1 extra
        appearances of each pid in `repeats`, any two appearances of the same pid at
        least `min_gap` positions apart. Pure function : writes nothing.

        Built by inserting the extra copies one at a time into a shuffled list, each at
        a random position that respects min_gap. An insertion only ever widens the gap
        between appearances already placed, so earlier constraints stay satisfied.
        """
        rng = np.random.default_rng(seed)
        for _ in range(max_tries) :
            order = list(rng.permutation(np.asarray(pids)))
            copies = [p for p in repeats for _ in range(n_repeats - 1)]
            ok = True
            for pid in rng.permutation(np.asarray(copies)) :
                placed = [j for j, p in enumerate(order) if p == pid]
                # Inserting at i : items at j < i stay at j, items at j >= i move to j + 1
                valid = [i for i in range(len(order) + 1)
                         if all((i - j if j < i else j + 1 - i) >= min_gap for j in placed)]
                if not valid :
                    ok = False
                    break
                order.insert(int(rng.choice(valid)), pid)
            if ok :
                return [str(p) for p in order]
        raise RuntimeError(f"Could not place the repeats with min_gap={min_gap} after {max_tries} tries")

    def _pids(self) -> list[str]:
        # Only the pid column is read : sids never leave master.csv
        return pd.read_csv(self.root / self.meta["master"]["file"], usecols=["pid"])["pid"].astype(str).tolist()

    def _write_json(self):
        # "w" : rewritten by add_rater and update_protocol
        with open(self.root / "project.json", "w") as f :
            json.dump(self.meta, f, indent=2)
            f.write("\n")

    @staticmethod
    def _hash(path : str | Path):
        with open(Path(path), 'rb') as f :
            hashed = hashlib.file_digest(f, "sha256").hexdigest()[:12]
        return hashed

    @classmethod
    def _check_hash(cls, path : str | Path, hashed : str):
        return cls._hash(path) == hashed


class AnnotationSession :
    """
    One rater working through their key. Answers go to one append-only CSV per rater,
    in long format (one row per item), keyed on `case` only : the pid and sid never
    appear in it, so the file does not reveal which cases are repeats.
    All rows written by one `save` share a `save_id` ; the latest save of a case wins,
    so re-saving a case corrects it without rewriting the file.
    """
    ANSWER_COLUMNS = ["case", "save_id", "item", "code", "quality", "comment",
                      "protocol_version", "seconds"]

    def __init__(self, project : AnnotationProject, rater : str, layout : dict = DEFAULT_LAYOUT) :
        if rater not in project.meta["raters"] :
            raise KeyError(f"Rater '{rater}' not registered : use AnnotationProject.add_rater. "
                           f"Registered : {list(project.meta['raters'])}")
        self.project, self.rater = project, rater
        self.info = project.meta["raters"][rater]
        self.protocol = project.protocol  # answers are always written with the current protocol
        self.answers_path = project.root / self.info["answers"]
        self.dataset = Dataset(project.meta["dataset_root"], "annotation", layout)
        self._key = self._load_key()
        self._sids = None  # pid -> sid, read from master.csv only when an image is needed
        self._answers = self._load_answers()

    def _load_key(self) -> pd.DataFrame :
        path = self.project.root / self.info["key"]
        # Checked again here : a session can be opened long after the project was loaded
        if not self.project._check_hash(path, self.info["sha256"]) :
            raise ValueError(f"Hash of the key of rater '{self.rater}' doesn't match project.json")
        key = pd.read_csv(path, dtype={"case" : int, "pid" : str})
        if list(key["case"]) != list(range(1, self.project.meta["n_cases"] + 1)) :
            raise ValueError(f"Key of rater '{self.rater}' does not hold cases 1..{self.project.meta['n_cases']}")
        return key.set_index("case")

    def _load_answers(self) -> dict :
        # case -> {item : {code, quality, comment, protocol_version, seconds}}, latest save only
        if not self.answers_path.exists() :
            return {}
        df = pd.read_csv(self.answers_path, dtype=str, keep_default_na=False)
        answers = {}
        for (case, save_id), rows in df.groupby(["case", "save_id"], sort=False) :
            # save_id is an ISO timestamp : string order is time order
            case = int(case)
            if case in answers and answers[case]["_save_id"] > save_id :
                continue
            answers[case] = {"_save_id" : save_id} | {
                r.item : dict(code=r.code, quality=r.quality, comment=r.comment,
                              protocol_version=int(r.protocol_version),
                              seconds=float(r.seconds) if r.seconds else None)
                for r in rows.itertuples()}
        return answers

    @property
    def order(self) -> list[int] :
        return list(self._key.index)

    def next_pending(self) -> int | None :
        # First case in presentation order without any saved answer, None when finished
        return next((c for c in self.order if c not in self._answers), None)

    def subject(self, case : int, modalities : list[str] | None = None) -> Subject :
        # The only place a sid is used : it stays inside the Subject, never displayed or written
        if self._sids is None :
            master = pd.read_csv(self.project.root / self.project.meta["master"]["file"], dtype=str)
            self._sids = dict(zip(master["pid"], master["sid"]))
        pid = self._key.loc[case, "pid"]
        return self.dataset.load_sub(self._sids[pid], modalities or ["t1w"], strict=True)

    def save(self, case : int, answers : dict, seconds : float | None = None) :
        """
        answers : {item : {"code" : ..., "quality" : ..., "comment" : ...}} for EVERY item
        of the current protocol. Everything is checked before anything is written.
        """
        case = int(case)
        if case not in self._key.index :
            raise KeyError(f"Case {case} is not in the key of rater '{self.rater}'")
        items = list(self.protocol.spec["items"])
        missing = [i for i in items if i not in answers]
        extra = [i for i in answers if i not in items]
        if missing or extra :
            raise ValueError(f"Case {case} : missing items {missing}, unknown items {extra}")
        for item in items :
            a = answers[item]
            self.protocol.check_answer(item, a["code"], a["quality"], a.get("comment", ""))

        save_id = datetime.now().isoformat(timespec="microseconds")
        rows = [dict(case=case, save_id=save_id, item=item, code=str(answers[item]["code"]),
                     quality=answers[item]["quality"], comment=answers[item].get("comment", "") or "",
                     protocol_version=self.protocol.version,
                     seconds="" if seconds is None else round(float(seconds), 1))
                for item in items]
        self.answers_path.parent.mkdir(exist_ok=True)
        new_file = not self.answers_path.exists()
        with open(self.answers_path, "a", newline="") as f :
            pd.DataFrame(rows, columns=self.ANSWER_COLUMNS).to_csv(f, header=new_file, index=False)
            f.flush()
            os.fsync(f.fileno())  # auto-save : the row is on disk before the next case opens
        self._answers[case] = {"_save_id" : save_id} | {
            r["item"] : dict(code=r["code"], quality=r["quality"], comment=r["comment"],
                             protocol_version=r["protocol_version"], seconds=seconds)
            for r in rows}

    def get(self, case : int) -> dict | None :
        a = self._answers.get(int(case))
        return None if a is None else {k : v for k, v in a.items() if k != "_save_id"}

    def progress(self) -> tuple[int, int] :
        return len(self._answers), len(self._key)
