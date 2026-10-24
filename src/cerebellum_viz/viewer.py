"""
napari front-ends : `view` displays one subject, `annotate` runs a rater's annotation session.
napari and magicgui are imported inside the functions, so the rest of the package works without them.
"""

from datetime import datetime
from pathlib import Path

from .annotation import AnnotationProject
from .data import DEFAULT_LAYOUT, Dataset, SES_LAYOUT

LAYOUTS = {"default" : DEFAULT_LAYOUT, "ses" : SES_LAYOUT}


def view(root : str | Path, sid : str, modalities : list[str] | None = None, ses : str | None = None,
         layout : dict | str = "default", mode : str = "2d", viewer=None) :
    """
    Open one subject of a dataset in napari. `layout` is a layout dict or a key of LAYOUTS
    ("ses" when the dataset has a session level). Missing modalities are skipped with a warning.
    Returns (viewer, subject).
    """
    import napari
    from napari.layers import Layer

    layout = LAYOUTS[layout] if isinstance(layout, str) else layout
    sub = Dataset(root, Path(root).name, layout).load_sub(sid, modalities, ses=ses)
    viewer = viewer or napari.Viewer(title=sid)
    for ld in sub.layers(mode) :
        viewer.add_layer(Layer.create(*ld))
    viewer.dims.ndisplay = 3 if mode == "3d" else 2
    viewer.reset_view()
    return viewer, sub


UNANSWERED = ""  # blank first entry : a case cannot be saved with a default left untouched

def annotate(project_root : str | Path, rater : str, modalities : list[str] | None = None,
             layout : dict = DEFAULT_LAYOUT, viewer=None) :
    """
    Open a napari viewer with the annotation form docked on the right, on the rater's
    first pending case. Blinding : the viewer only ever shows "case i / N" and layer
    names such as "t1w" ; no sid or pid is displayed.
    """
    # Imported here so the rest of the module works without napari installed
    import time
    import napari
    from napari.layers import Layer
    from magicgui.widgets import ComboBox, Container, Label, LineEdit, PushButton, SpinBox

    session = AnnotationProject.load(project_root).session(rater, layout)
    protocol = session.protocol
    order = session.order
    viewer = viewer or napari.Viewer(title=f"Annotation - {rater}")

    # ---- Widgets, built from the protocol ----
    header = Label(value="")
    status = Label(value="")
    fields = {}
    rows = []
    for item, spec in protocol.spec["items"].items() :
        codes = {**spec["choices"], **protocol.spec["extra_choices"]}
        code_box = ComboBox(label=spec["label"],
                            choices=[("-- choose --", UNANSWERED)] + [(f"{c} : {t}", c) for c, t in codes.items()])
        code_box.tooltip = spec["description"]
        quality_box = ComboBox(label="quality",
                               choices=[(f"{q} : {t}", q) for q, t in protocol.spec["quality"].items()])
        comment = LineEdit(label="comment")
        fields[item] = (code_box, quality_box, comment)
        rows += [Label(value=f"<b>{item}</b>"), code_box, quality_box, comment]

    goto = SpinBox(label="go to case", min=1, max=len(order), value=1)
    btn_prev, btn_next = PushButton(text="< previous"), PushButton(text="next >")
    btn_save = PushButton(text="Save and next pending")
    btn_goto = PushButton(text="go")
    form = Container(widgets=[header, *rows, btn_save,
                              Container(widgets=[btn_prev, btn_next], layout="horizontal", labels=False),
                              Container(widgets=[goto, btn_goto], layout="horizontal"),
                              status])
    state = {"case" : None, "t0" : None, "ok" : False}

    # ---- Case display ----
    def open_case(case) :
        viewer.layers.clear()
        sub = session.subject(case, modalities)
        for ld in sub.layers("2d") :
            viewer.add_layer(Layer.create(*ld))
        # Axis 0 is left-right (RSA) : start on the middle sagittal slice
        viewer.dims.ndisplay = 2
        viewer.dims.set_current_step(0, viewer.layers[0].data.shape[0] // 2)
        viewer.reset_view()

        previous = session.get(case)
        for item, (code_box, quality_box, comment) in fields.items() :
            a = (previous or {}).get(item)
            if a :
                code_box.value = a["code"]
                quality_box.value = a["quality"]
            else :
                code_box.value = UNANSWERED
                quality_box.value = next(iter(protocol.spec["quality"]))
            comment.value = a["comment"] if a else ""

        state["t0"] = time.monotonic()
        goto.value = case
        done, total = session.progress()
        header.value = f"<b>Case {case} / {total}</b>  -  {done} done"
        status.value = "already annotated : saving again replaces the answer" if previous else ""

    def show(case) :
        # The case is set first so that navigation still works from a case that failed to open
        state["case"], state["ok"] = case, False
        try :
            open_case(case)
            state["ok"] = True
        except Exception as e :  # a missing image must not kill the session
            viewer.layers.clear()
            for code_box, _, comment in fields.values() :
                code_box.value, comment.value = UNANSWERED, ""
            header.value = f"<b>Case {case} / {len(order)}</b>"
            # The error text holds the image path, hence the sid : it goes to a log, not on screen
            with open(session.project.root / "errors.log", "a") as log :
                log.write(f"{datetime.now().isoformat(timespec='seconds')} {rater} case {case} : {e!r}\n")
            status.value = f"cannot open case {case} ({type(e).__name__}) : details in errors.log, case not saved"

    # ---- Actions ----
    def save_and_next() :
        case = state["case"]
        if not state["ok"] :
            status.value = f"not saved : case {case} is not displayed"
            return
        answers = {item : dict(code=cb.value, quality=qb.value, comment=cm.value.strip())
                   for item, (cb, qb, cm) in fields.items()}
        blank = [item for item, a in answers.items() if a["code"] == UNANSWERED]
        if blank :
            status.value = f"not saved : no answer for {blank}"
            return
        try :
            session.save(case, answers, seconds=time.monotonic() - state["t0"])
        except (ValueError, KeyError) as e :
            status.value = f"not saved : {e}"
            return
        nxt = session.next_pending()
        if nxt is None :
            header.value = f"<b>All {len(order)} cases done</b>"
            status.value = f"case {case} saved"
            return
        show(nxt)
        status.value = f"case {case} saved"

    def step(delta) :
        i = order.index(state["case"]) + delta
        if 0 <= i < len(order) :
            show(order[i])

    btn_save.changed.connect(save_and_next)
    btn_prev.changed.connect(lambda : step(-1))
    btn_next.changed.connect(lambda : step(+1))
    btn_goto.changed.connect(lambda : show(goto.value))

    viewer.window.add_dock_widget(form, name="Annotation", area="right")
    first = session.next_pending()
    if first is None :
        show(order[0])
        header.value = f"<b>All {len(order)} cases done</b>"
    else :
        show(first)
    return viewer, session, form
