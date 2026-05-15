"""Synthetic ARC-style data generation for CoE-LIFT.

The generator is deliberately small and inspectable. It creates paired multilingual
views of the same abstract rule instance so CoE alignment has well-defined positives.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

Grid = list[list[int]]

TRAIN_LANGS = ("en", "zh", "es", "fr")
OOD_LANGS = ("ar", "hi", "sw", "ta", "bn")


LANG_TEMPLATES = {
    "en": {
        "instruction": (
            "Infer the hidden grid transformation from the examples. "
            "Return only the final answer as JSON with key \"grid\"."
        ),
        "example": "Example {i}\nInput: {inp}\nOutput: {out}",
        "test": "Test\nInput: {inp}\nAnswer:",
    },
    "zh": {
        "instruction": "根据示例推断隐藏的网格变换。只返回包含 \"grid\" 键的 JSON。",
        "example": "示例 {i}\n输入: {inp}\n输出: {out}",
        "test": "测试\n输入: {inp}\n答案:",
    },
    "es": {
        "instruction": (
            "Infiere la transformación oculta de la cuadrícula a partir de los ejemplos. "
            "Devuelve solo JSON con la clave \"grid\"."
        ),
        "example": "Ejemplo {i}\nEntrada: {inp}\nSalida: {out}",
        "test": "Prueba\nEntrada: {inp}\nRespuesta:",
    },
    "fr": {
        "instruction": (
            "Déduis la transformation cachée de la grille à partir des exemples. "
            "Retourne uniquement du JSON avec la clé \"grid\"."
        ),
        "example": "Exemple {i}\nEntrée: {inp}\nSortie: {out}",
        "test": "Test\nEntrée: {inp}\nRéponse:",
    },
    "ar": {
        "instruction": "استنتج تحويل الشبكة المخفي من الأمثلة. أعد فقط JSON بالمفتاح \"grid\".",
        "example": "مثال {i}\nالمدخل: {inp}\nالمخرج: {out}",
        "test": "اختبار\nالمدخل: {inp}\nالإجابة:",
    },
    "hi": {
        "instruction": "उदाहरणों से छिपा हुआ grid परिवर्तन निकालें। केवल \"grid\" कुंजी वाला JSON लौटाएं।",
        "example": "उदाहरण {i}\nइनपुट: {inp}\nआउटपुट: {out}",
        "test": "परीक्षण\nइनपुट: {inp}\nउत्तर:",
    },
    "sw": {
        "instruction": "Tambua mabadiliko yaliyofichwa ya gridi kutoka kwenye mifano. Rudisha JSON tu yenye ufunguo \"grid\".",
        "example": "Mfano {i}\nIngizo: {inp}\nTokeo: {out}",
        "test": "Jaribio\nIngizo: {inp}\nJibu:",
    },
    "ta": {
        "instruction": "எடுத்துக்காட்டுகளிலிருந்து மறைந்த grid மாற்றத்தை கண்டறிக. \"grid\" விசையுடன் JSON மட்டும் திருப்பு.",
        "example": "எடுத்துக்காட்டு {i}\nஉள்ளீடு: {inp}\nவெளியீடு: {out}",
        "test": "சோதனை\nஉள்ளீடு: {inp}\nபதில்:",
    },
    "bn": {
        "instruction": "উদাহরণ থেকে লুকানো grid রূপান্তরটি অনুমান করুন। শুধু \"grid\" কীসহ JSON ফেরত দিন।",
        "example": "উদাহরণ {i}\nইনপুট: {inp}\nআউটপুট: {out}",
        "test": "পরীক্ষা\nইনপুট: {inp}\nউত্তর:",
    },
}


SINGLE_RULES = ("rotate_cw", "reflect_h", "shift_right", "color_map", "count_nonzero")
TRAIN_RULES = (
    ("rotate_cw",),
    ("reflect_h",),
    ("shift_right",),
    ("color_map",),
    ("rotate_cw", "color_map"),
    ("reflect_h", "shift_right"),
    ("shift_right", "color_map"),
)
OOD_RULES = (
    ("rotate_cw", "color_map", "reflect_h"),
    ("reflect_h", "shift_right", "color_map"),
    ("shift_right", "rotate_cw", "color_map"),
    ("color_map", "reflect_h", "count_nonzero"),
)


@dataclass(frozen=True)
class GeneratedGroup:
    group_id: int
    rule_seq: tuple[str, ...]
    examples: tuple[tuple[Grid, Grid], ...]
    test_input: Grid
    test_output: Grid
    size: int


def canonical_answer(grid: Grid) -> str:
    return json.dumps({"grid": grid}, ensure_ascii=False, separators=(",", ":"))


def grid_to_text(grid: Grid) -> str:
    return json.dumps(grid, ensure_ascii=False, separators=(",", ":"))


def rotate_cw(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]


def reflect_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def shift_right(grid: Grid) -> Grid:
    return [[row[-1], *row[:-1]] for row in grid]


def color_map(grid: Grid) -> Grid:
    mapping = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 1, 6: 7, 7: 8, 8: 9, 9: 6}
    return [[mapping.get(v, v) for v in row] for row in grid]


def count_nonzero(grid: Grid) -> Grid:
    count = sum(1 for row in grid for value in row if value != 0)
    return [[count % 10]]


RULES: dict[str, Callable[[Grid], Grid]] = {
    "rotate_cw": rotate_cw,
    "reflect_h": reflect_h,
    "shift_right": shift_right,
    "color_map": color_map,
    "count_nonzero": count_nonzero,
}


def apply_rules(grid: Grid, rule_seq: Iterable[str]) -> Grid:
    out = [row[:] for row in grid]
    for rule in rule_seq:
        out = RULES[rule](out)
    return out


def make_grid(rng: random.Random, size: int, max_color: int = 5) -> Grid:
    grid = [[0 for _ in range(size)] for _ in range(size)]
    n_cells = rng.randint(max(2, size // 2), max(3, size + 2))
    for _ in range(n_cells):
        r = rng.randrange(size)
        c = rng.randrange(size)
        grid[r][c] = rng.randint(1, max_color)

    if rng.random() < 0.55 and size >= 4:
        h = rng.randint(1, min(3, size))
        w = rng.randint(1, min(3, size))
        r0 = rng.randint(0, size - h)
        c0 = rng.randint(0, size - w)
        color = rng.randint(1, max_color)
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                grid[r][c] = color
    return grid


def permute_surface(grid: Grid, seed: int) -> Grid:
    if seed == 0:
        return [row[:] for row in grid]
    rng = random.Random(seed)
    values = list(range(1, 10))
    shuffled = values[:]
    rng.shuffle(shuffled)
    mapping = {0: 0, **dict(zip(values, shuffled))}
    return [[mapping.get(v, v) for v in row] for row in grid]


def format_prompt(lang: str, examples: tuple[tuple[Grid, Grid], ...], test_input: Grid) -> str:
    tmpl = LANG_TEMPLATES[lang]
    parts = [tmpl["instruction"]]
    for i, (inp, out) in enumerate(examples, start=1):
        parts.append(tmpl["example"].format(i=i, inp=grid_to_text(inp), out=grid_to_text(out)))
    parts.append(tmpl["test"].format(inp=grid_to_text(test_input)))
    return "\n\n".join(parts)


def make_group(
    group_id: int,
    rng: random.Random,
    rule_seq: tuple[str, ...],
    size_range: tuple[int, int],
    examples_per_task: int,
) -> GeneratedGroup:
    size = rng.randint(*size_range)
    inputs = [make_grid(rng, size) for _ in range(examples_per_task + 1)]
    outputs = [apply_rules(grid, rule_seq) for grid in inputs]
    examples = tuple(zip(inputs[:-1], outputs[:-1]))
    return GeneratedGroup(
        group_id=group_id,
        rule_seq=rule_seq,
        examples=examples,
        test_input=inputs[-1],
        test_output=outputs[-1],
        size=size,
    )


def make_record(
    group: GeneratedGroup,
    lang: str,
    split: str,
    surface_seed: int,
    record_index: int,
    ood_label: int,
) -> dict[str, object]:
    examples = tuple(
        (permute_surface(inp, surface_seed), permute_surface(out, surface_seed))
        for inp, out in group.examples
    )
    test_input = permute_surface(group.test_input, surface_seed)
    test_output = permute_surface(group.test_output, surface_seed)
    return {
        "id": f"task_{group.group_id:06d}_{lang}_perm{surface_seed}_{record_index}",
        "group_id": group.group_id,
        "lang": lang,
        "split": split,
        "prompt": format_prompt(lang, examples, test_input),
        "answer": canonical_answer(test_output),
        "rule_family": "+".join(group.rule_seq),
        "surface_seed": surface_seed,
        "ood_label": int(ood_label),
        "grid_size": group.size,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def generate_coe_lift_dataset(
    output_dir: str | Path,
    train_groups: int = 1200,
    eval_groups: int = 300,
    examples_per_task: int = 3,
    seed: int = 13,
    include_probe_shifts: bool = True,
) -> dict[str, object]:
    """Generate train/eval JSONL files and return metadata."""

    output_dir = Path(output_dir)
    rng = random.Random(seed)
    train_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    group_id = 0
    record_index = 0

    for _ in range(train_groups):
        group_id += 1
        rule_seq = rng.choice(TRAIN_RULES)
        group = make_group(group_id, rng, rule_seq, (3, 7), examples_per_task)
        for lang in TRAIN_LANGS:
            for surface_seed in (0, 1):
                record_index += 1
                train_rows.append(
                    make_record(group, lang, "train_id", surface_seed, record_index, 0)
                )
        if include_probe_shifts:
            for lang in rng.sample(TRAIN_LANGS, k=2):
                record_index += 1
                train_rows.append(
                    make_record(group, lang, "train_probe_surface", 1000 + group_id, record_index, 1)
                )

    eval_buckets = (
        ("test_id", TRAIN_LANGS, TRAIN_RULES, (3, 7), (0,), 0),
        ("test_ood_lang", OOD_LANGS, TRAIN_RULES, (3, 7), (0,), 1),
        ("test_ood_rule", TRAIN_LANGS, OOD_RULES, (3, 7), (0,), 1),
        ("test_ood_surface", TRAIN_LANGS, TRAIN_RULES, (8, 12), (7000, 7001), 1),
    )
    per_bucket = max(1, eval_groups // len(eval_buckets))

    for split, langs, rules, size_range, surface_seeds, ood_label in eval_buckets:
        for _ in range(per_bucket):
            group_id += 1
            rule_seq = rng.choice(rules)
            group = make_group(group_id, rng, rule_seq, size_range, examples_per_task)
            selected_langs = langs if len(langs) <= 4 else rng.sample(list(langs), k=4)
            for lang in selected_langs:
                for surface_seed in surface_seeds:
                    record_index += 1
                    eval_rows.append(
                        make_record(group, lang, split, surface_seed, record_index, ood_label)
                    )

    train_path = output_dir / "coe_lift_train.jsonl"
    eval_path = output_dir / "coe_lift_eval.jsonl"
    train_count = write_jsonl(train_path, train_rows)
    eval_count = write_jsonl(eval_path, eval_rows)

    metadata = {
        "seed": seed,
        "train_rows": train_count,
        "eval_rows": eval_count,
        "train_groups": train_groups,
        "eval_groups_requested": eval_groups,
        "examples_per_task": examples_per_task,
        "train_langs": list(TRAIN_LANGS),
        "ood_langs": list(OOD_LANGS),
        "train_rules": ["+".join(rule) for rule in TRAIN_RULES],
        "ood_rules": ["+".join(rule) for rule in OOD_RULES],
        "paths": {
            "train_jsonl": str(train_path),
            "eval_jsonl": str(eval_path),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
