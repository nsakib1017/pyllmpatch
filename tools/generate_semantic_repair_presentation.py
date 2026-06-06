from __future__ import annotations

import json
import statistics
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "semantic_repair_workflow_results.pptx"

SLIDE_W = 12_192_000
SLIDE_H = 6_858_000

NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def emu(inches: float) -> int:
    return int(round(inches * 914_400))


def read_json_objects(path: Path) -> list[dict[str, Any]]:
    """Read JSONL files that may contain occasional concatenated JSON objects."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    rows: list[dict[str, Any]] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        if isinstance(obj, dict):
            rows.append(obj)
        idx = end
    return rows


def pct(num: float, den: float, digits: int = 1) -> str:
    if not den:
        return "0%"
    return f"{num / den * 100:.{digits}f}%"


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def collect_metrics() -> dict[str, Any]:
    result_rows = read_json_objects(REPO_ROOT / "results.jsonl")
    raw = read_json_objects(REPO_ROOT / "dataset" / "accepted_codeobject_mining_raw.jsonl")
    refreshed = read_json_objects(REPO_ROOT / "dataset" / "accepted_codeobject_mining_prompt_refresh.jsonl")

    completed = [row for row in result_rows if row.get("stage") == "result"]
    skipped = [row for row in result_rows if row.get("stage") == "timeout"]
    paired = [
        row
        for row in completed
        if isinstance(row.get("initial_combined_distance"), (int, float))
        and isinstance(row.get("final_combined_distance"), (int, float))
    ]
    improved = [row for row in paired if row["final_combined_distance"] < row["initial_combined_distance"]]
    unchanged = [row for row in paired if row["final_combined_distance"] == row["initial_combined_distance"]]
    worse = [row for row in paired if row["final_combined_distance"] > row["initial_combined_distance"]]
    exact = [row for row in paired if row["final_combined_distance"] == 0]
    initial_sum = sum(row["initial_combined_distance"] for row in paired)
    final_sum = sum(row["final_combined_distance"] for row in paired)
    timed_partial = [row for row in completed if row.get("sample_timed_out") is True]

    raw_models = Counter(
        (row.get("prompt") or {}).get("model") if isinstance(row.get("prompt"), dict) else None
        for row in raw
    )
    raw_with_prompt = sum(1 for row in raw if isinstance(row.get("prompt"), dict))
    refresh_strategy = Counter(
        (row.get("prompt") or {}).get("candidate_strategy")
        if isinstance(row.get("prompt"), dict)
        else None
        for row in refreshed
    )

    return {
        "total_records": len(result_rows),
        "completed_count": len(completed),
        "timeout_count": len(skipped),
        "partial_timeout_count": len(timed_partial),
        "exact_count": len(exact),
        "partial_count": len(completed) - len(exact),
        "improved_count": len(improved),
        "unchanged_count": len(unchanged),
        "worse_count": len(worse),
        "initial_sum": initial_sum,
        "final_sum": final_sum,
        "distance_reduction": initial_sum - final_sum,
        "distance_reduction_pct": pct(initial_sum - final_sum, initial_sum),
        "median_final_distance": statistics.median([row["final_combined_distance"] for row in paired]),
        "median_delta": statistics.median(
            [row["initial_combined_distance"] - row["final_combined_distance"] for row in paired]
        ),
        "accepted_steps": sum((row.get("accepted_step_count") or 0) for row in completed),
        "median_accepted_steps": statistics.median([(row.get("accepted_step_count") or 0) for row in completed]),
        "raw_accepted_rows": len(raw),
        "raw_source_files": len({row.get("source_file_hash") for row in raw}),
        "raw_runs": len({row.get("run_id") for row in raw}),
        "raw_prompt_coverage": pct(raw_with_prompt, len(raw), digits=2),
        "raw_models": raw_models,
        "refresh_count": len(refreshed),
        "refresh_strategy": refresh_strategy,
    }


def rpr(size: int, color: str, bold: bool = False) -> str:
    b = ' b="1"' if bold else ""
    return (
        f'<a:rPr lang="en-US" sz="{size}"{b}>'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        '<a:latin typeface="Aptos"/>'
        "</a:rPr>"
    )


def paragraph(text: str, *, size: int, color: str, bold: bool = False, bullet: bool = False) -> str:
    clean = escape(text)
    if bullet:
        ppr = (
            f'<a:pPr marL="{emu(0.25)}" indent="-{emu(0.13)}">'
            '<a:buChar char="&#8226;"/>'
            f'<a:defRPr sz="{size}"/>'
            "</a:pPr>"
        )
    else:
        ppr = f'<a:pPr><a:defRPr sz="{size}"/></a:pPr>'
    return f"<a:p>{ppr}<a:r>{rpr(size, color, bold)}<a:t>{clean}</a:t></a:r></a:p>"


def shape(
    sid: int,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    paras: list[str],
    *,
    fill: str | None = None,
    line: str | None = None,
) -> str:
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
    line_xml = f'<a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
    return f"""
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="{sid}" name="{escape(name)}"/>
          <p:cNvSpPr txBox="1"/>
          <p:nvPr/>
        </p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          {fill_xml}
          {line_xml}
        </p:spPr>
        <p:txBody>
          <a:bodyPr wrap="square" rtlCol="0"><a:spAutoFit/></a:bodyPr>
          <a:lstStyle/>
          {''.join(paras)}
        </p:txBody>
      </p:sp>"""


def slide_xml(slide: dict[str, Any]) -> str:
    items: list[str] = []
    sid = 2
    items.append(
        shape(
            sid,
            "Accent",
            0,
            0,
            13.333,
            0.16,
            [paragraph("", size=100, color="FFFFFF")],
            fill="2F6F73",
        )
    )
    sid += 1
    items.append(
        shape(
            sid,
            "Title",
            0.62,
            0.34,
            12.0,
            0.62,
            [paragraph(slide["title"], size=3000, color="17202A", bold=True)],
        )
    )
    sid += 1
    if slide.get("subtitle"):
        items.append(
            shape(
                sid,
                "Subtitle",
                0.65,
                0.96,
                11.7,
                0.42,
                [paragraph(slide["subtitle"], size=1250, color="52616B")],
            )
        )
        sid += 1
    if slide.get("cards"):
        x = 0.65
        for label, value, color in slide["cards"]:
            items.append(
                shape(
                    sid,
                    f"Metric {label}",
                    x,
                    1.45,
                    2.33,
                    0.88,
                    [
                        paragraph(str(value), size=2250, color=color, bold=True),
                        paragraph(label, size=900, color="4D5B63"),
                    ],
                    fill="FFFFFF",
                    line="D7DEE2",
                )
            )
            sid += 1
            x += 2.48
    body_y = slide.get("body_y", 1.45 if not slide.get("cards") else 2.55)
    body_h = 5.2 if not slide.get("cards") else 3.9
    body_paras = [
        paragraph(item, size=1550, color="202A33", bullet=True)
        for item in slide["bullets"]
    ]
    items.append(shape(sid, "Bullets", 0.82, body_y, 11.6, body_h, body_paras))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="F6F8FA"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr>
      {''.join(items)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def relationships(entries: list[tuple[str, str, str]]) -> str:
    rows = "\n".join(
        f'  <Relationship Id="{rid}" Type="{rtype}" Target="{escape(target)}"/>'
        for rid, rtype, target in entries
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{REL_NS}">
{rows}
</Relationships>"""


def content_types(slide_count: int) -> str:
    overrides = [
        "/ppt/presentation.xml|application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml",
        "/ppt/slideMasters/slideMaster1.xml|application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml",
        "/ppt/slideLayouts/slideLayout1.xml|application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml",
        "/ppt/theme/theme1.xml|application/vnd.openxmlformats-officedocument.theme+xml",
        "/docProps/core.xml|application/vnd.openxmlformats-package.core-properties+xml",
        "/docProps/app.xml|application/vnd.openxmlformats-officedocument.extended-properties+xml",
    ]
    overrides.extend(
        f"/ppt/slides/slide{i}.xml|application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
        for i in range(1, slide_count + 1)
    )
    override_xml = "\n".join(
        f'  <Override PartName="{part}" ContentType="{ctype}"/>'
        for item in overrides
        for part, ctype in [item.split("|", 1)]
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
{override_xml}
</Types>"""


def presentation_xml(slide_count: int) -> str:
    sld_ids = "\n".join(
        f'    <p:sldId id="{255 + i}" r:id="rId{i + 1}"/>'
        for i in range(1, slide_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>
{sld_ids}
  </p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle>
</p:presentation>"""


def slide_master_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""


def slide_layout_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def theme_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="{NS_A}" name="Pyllmpatch">
  <a:themeElements>
    <a:clrScheme name="Pyllmpatch">
      <a:dk1><a:srgbClr val="17202A"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="202A33"/></a:dk2><a:lt2><a:srgbClr val="F6F8FA"/></a:lt2>
      <a:accent1><a:srgbClr val="2F6F73"/></a:accent1><a:accent2><a:srgbClr val="D95F43"/></a:accent2>
      <a:accent3><a:srgbClr val="3A7D44"/></a:accent3><a:accent4><a:srgbClr val="7B5EA7"/></a:accent4>
      <a:accent5><a:srgbClr val="C08A2B"/></a:accent5><a:accent6><a:srgbClr val="4D6B8A"/></a:accent6>
      <a:hlink><a:srgbClr val="2F6F73"/></a:hlink><a:folHlink><a:srgbClr val="7B5EA7"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Pyllmpatch"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def props_xml(slide_count: int) -> tuple[str, str]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Semantic Repair Workflow and Results</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""
    app = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{slide_count}</Slides>
</Properties>"""
    return core, app


def build_slides(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    models = metrics["raw_models"]
    strategies = metrics["refresh_strategy"]
    return [
        {
            "title": "Semantic Code-Object Repair Workflow",
            "subtitle": "Brief workflow for verifier-gated repair of Python decompiler semantic drift",
            "bullets": [
                "Start from decompiler output whose recompiled .pyc differs from the ground-truth .pyc.",
                "Map source fragments to Python code objects and select the failed PyLingual target.",
                "Build a localized prompt from target metadata, bytecode diff, source fragment, and edit contracts.",
                "Generate candidate source-fragment replacements with the repair model.",
                "Reattach, compile, and score each candidate using combined code-object distance.",
                "Accept only candidates that improve the verified distance; iterate until exact match, no improvement, or timeout.",
            ],
        },
        {
            "title": "Prompt Components",
            "subtitle": "The prompt is structured to keep repairs local and bytecode-grounded",
            "bullets": [
                "System role: bytecode-aware semantic repair specialist; output source fragment only.",
                "Task and target qualname: distinguishes module statement repair, fragment repair, or missing insertion.",
                "Indentation contract: preserve first-line indentation, return a complete replacement, omit line prefixes.",
                "Line-numbered source fragment: the only editable code span, with source location hints.",
                "Failure context and instruction diff: PyLingual failure, alignment tag, localized gt/derived bytecode windows.",
                "Code-object metadata: compact identical/different co_* fields for names, constants, varnames, flags, and closures.",
                "Optional guidance: rejected attempts, candidate strategy, prior accepted telemetry, and mined action patterns.",
            ],
        },
        {
            "title": "Fine-Tuning Dataset: Accepted Code Objects",
            "subtitle": "Teacher responses are kept only when the verifier accepted the repair",
            "cards": [
                ("raw accepted rows", fmt_int(metrics["raw_accepted_rows"]), "2F6F73"),
                ("source files", fmt_int(metrics["raw_source_files"]), "3A7D44"),
                ("refreshed samples", fmt_int(metrics["refresh_count"]), "D95F43"),
                ("prompt coverage", metrics["raw_prompt_coverage"], "7B5EA7"),
            ],
            "bullets": [
                f"Mined {fmt_int(metrics['raw_accepted_rows'])} accepted repair records from {metrics['raw_runs']} larger-model semantic repair runs.",
                f"Teacher mix: {fmt_int(models.get('gpt-5.4-mini', 0))} GPT-5.4-mini responses and {fmt_int(models.get('gpt-5.5-1', 0))} GPT-5.5-1 responses.",
                "Each label is the accepted replacement_text, not an unverified raw model answer.",
                "Acceptance criterion: reattach candidate, compile it, and keep it only if combined code-object distance improved.",
                "Offline prompt refresh rebuilds records with the current semantic_v4 template while preserving the accepted output.",
                f"Expansion uses candidate strategies: {fmt_int(strategies.get('retrieval_guided', 0))} retrieval-guided, {fmt_int(strategies.get('control_flow_residual', 0))} control-flow, {fmt_int(strategies.get('call_attribute_residual', 0))} call/attribute samples.",
            ],
        },
        {
            "title": "results.jsonl Analysis",
            "subtitle": "Parsed local results.jsonl; one physical line contains two JSON objects",
            "cards": [
                ("parsed records", fmt_int(metrics["total_records"]), "2F6F73"),
                ("result rows", fmt_int(metrics["completed_count"]), "3A7D44"),
                ("exact matches", fmt_int(metrics["exact_count"]), "D95F43"),
                ("regressions", fmt_int(metrics["worse_count"]), "7B5EA7"),
            ],
            "bullets": [
                f"{fmt_int(metrics['completed_count'])} completed result rows and {fmt_int(metrics['timeout_count'])} timeout/no-improvement rows.",
                f"Completed rows: {fmt_int(metrics['exact_count'])} exact, {fmt_int(metrics['partial_count'])} partial; exact rate = {pct(metrics['exact_count'], metrics['completed_count'])}.",
                f"Distance movement: {fmt_int(metrics['improved_count'])} improved, {fmt_int(metrics['unchanged_count'])} unchanged, {fmt_int(metrics['worse_count'])} worse.",
                f"Combined distance sum fell from {fmt_int(metrics['initial_sum'])} to {fmt_int(metrics['final_sum'])}: -{fmt_int(metrics['distance_reduction'])} ({metrics['distance_reduction_pct']}).",
                f"Median final distance is {metrics['median_final_distance']}; median accepted improvement is {fmt_int(metrics['median_delta'])}.",
                f"{fmt_int(metrics['accepted_steps'])} accepted repair steps across completed rows; median {metrics['median_accepted_steps']} accepted step per sample.",
            ],
        },
        {
            "title": "Interpretation",
            "subtitle": "What the results suggest for the next repair iteration",
            "bullets": [
                "Verifier-gated acceptance is high precision: no completed sample increased combined distance.",
                "Exact-match rate is strong on completed rows, but many samples stop before useful improvement.",
                f"Timeout pressure is the main bottleneck: {fmt_int(metrics['timeout_count'])} no-improvement skips plus {fmt_int(metrics['partial_timeout_count'])} partial results kept after timeout.",
                "Mean distance is skewed by high-distance outliers; median final distance and exact-match rate are more informative.",
                "The accepted-code-object dataset should help the local model learn small bytecode-aligned edits and reduce teacher-model calls.",
                "Next focus: target timeout-heavy cases with better retrieval, candidate diversity, and residual-specific prompts.",
            ],
        },
    ]


def write_pptx(slides: list[dict[str, Any]], path: Path) -> None:
    core, app = props_xml(len(slides))
    presentation_rels = [("rId1", f"{NS_R}/slideMaster", "slideMasters/slideMaster1.xml")]
    presentation_rels.extend(
        (f"rId{i + 1}", f"{NS_R}/slide", f"slides/slide{i}.xml")
        for i in range(1, len(slides) + 1)
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(slides)))
        zf.writestr(
            "_rels/.rels",
            relationships(
                [
                    ("rId1", f"{NS_R}/officeDocument", "ppt/presentation.xml"),
                    ("rId2", f"{NS_R}/metadata/core-properties", "docProps/core.xml"),
                    ("rId3", f"{NS_R}/extended-properties", "docProps/app.xml"),
                ]
            ),
        )
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", relationships(presentation_rels))
        zf.writestr("ppt/theme/theme1.xml", theme_xml())
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        zf.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            relationships(
                [
                    ("rId1", f"{NS_R}/slideLayout", "../slideLayouts/slideLayout1.xml"),
                    ("rId2", f"{NS_R}/theme", "../theme/theme1.xml"),
                ]
            ),
        )
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        zf.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            relationships([("rId1", f"{NS_R}/slideMaster", "../slideMasters/slideMaster1.xml")]),
        )
        for idx, slide in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide_xml(slide))
            zf.writestr(
                f"ppt/slides/_rels/slide{idx}.xml.rels",
                relationships([("rId1", f"{NS_R}/slideLayout", "../slideLayouts/slideLayout1.xml")]),
            )


def main() -> None:
    metrics = collect_metrics()
    slides = build_slides(metrics)
    write_pptx(slides, OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    print(json.dumps({key: value for key, value in metrics.items() if not isinstance(value, Counter)}, indent=2))


if __name__ == "__main__":
    main()
