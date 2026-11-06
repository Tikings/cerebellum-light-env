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
MIDDLE = 3  # napari's mouse button numbers : 1 left, 2 right, 3 middle

def annotate(project_root : str | Path, rater : str, modalities : list[str] | None = None,
             layout : dict = DEFAULT_LAYOUT, viewer=None) :
    """
    Open a napari viewer with the annotation form docked on the right, on the rater's
    first pending case. Blinding : the viewer only ever shows "case i / N" and layer
    names such as "t1w" ; no sid or pid is displayed.
    With a protocol that has landmarks (v2), the codes chosen show exactly the landmarks they
    need (protocol `required`), parked in a labelled column left of the image ; the only action
    on them is to drag them into place, one at a time. They follow the displayed sagittal slice :
    the slice they are saved on is the case's mid-sagittal slice. napari's layer panels are
    hidden (contrast is in the form), and anything that adds, removes or multi-selects points
    is undone.
    """
    # Imported here so the rest of the module works without napari installed
    import time
    import numpy as np
    import pandas as pd
    import napari
    from napari.layers import Layer
    from magicgui.widgets import (CheckBox, ComboBox, Container, FloatRangeSlider, Label, LineEdit,
                                  PushButton, SpinBox)
    from qtpy.QtCore import Qt, QTimer
    from qtpy.QtWidgets import QComboBox

    session = AnnotationProject.load(project_root).session(rater, layout)
    protocol = session.protocol
    order = session.order
    viewer = viewer or napari.Viewer(title=f"Annotation - {rater}")

    def middle_pan(viewer, event) :
        # Middle-button drag pans the image in any layer mode (napari only pans in pan_zoom mode,
        # and with the left button) : the world point grabbed at press stays under the cursor
        if event.button != MIDDLE :
            return
        grab = np.array(event.position)[list(event.dims_displayed)]
        yield
        while event.type == "mouse_move" :
            here = np.array(event.position)[list(event.dims_displayed)]
            center = np.array(viewer.camera.center, dtype=float)
            center[-len(grab):] -= here - grab
            viewer.camera.center = tuple(center)
            yield

    viewer.mouse_drag_callbacks.append(middle_pan)

    # ---- Widgets, built from the protocol ----
    header = Label(value="")
    status = Label(value="")
    fields = {}
    tick_boxes = {}  # item -> {tick : CheckBox} (protocol >= 2)
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
        tick_boxes[item] = {t : CheckBox(text=ts["label"], value=False, enabled=False)
                            for t, ts in protocol.ticks(item).items()}
        rows += [Label(value=f"<b>{item}</b>"), code_box, *tick_boxes[item].values(), quality_box, comment]

    def item_code(item) :
        # The code saved for an item : the chosen code, plus its ticks ("2+bifid1")
        base = fields[item][0].value
        if base == UNANSWERED :
            return UNANSWERED
        return protocol.join_code(item, base, {t for t, cb in tick_boxes[item].items() if cb.value})

    def update_ticks(item) :
        # A tick is offered only for the codes it applies to (folium 2 bifid : 2 folia only)
        base = fields[item][0].value
        for t, cb in tick_boxes[item].items() :
            allowed = base in protocol.ticks(item)[t]["for"]
            cb.enabled = allowed
            if not allowed :
                cb.value = False

    # ---- Landmark widgets (protocol >= 2 only) ----
    landmarks = protocol.landmarks
    lm_order = list(landmarks)
    lm_rows = []
    if landmarks :
        lm_status = Label(value="")
        contrast = FloatRangeSlider(label="contrast", min=0, max=1, value=(0, 1))
        btn_reset = PushButton(text="reset view")
        lm_rows = [Label(value="<b>landmarks</b>"),
                   Label(value="<small>The codes show the landmarks they need, parked left of the image. "
                               "Drag each one into place on the mid-sagittal slice : the landmarks "
                               "follow the slice shown. Scroll : zoom ; middle-drag or hold Space : pan. "
                               "N : names on/off ; H : landmarks on/off (hidden landmarks "
                               "cannot be dragged).</small>"),
                   lm_status, contrast, btn_reset]

    goto = SpinBox(label="go to case", min=1, max=len(order), value=1)
    btn_prev, btn_next = PushButton(text="< previous"), PushButton(text="next >")
    btn_save = PushButton(text="Save and next pending")
    btn_goto = PushButton(text="go")
    # scrollable : the form is taller than a laptop screen ; without a scroll area its height
    # becomes the window's minimum and pushes the slice slider and buttons off screen
    form = Container(widgets=[header, *rows, *lm_rows, btn_save,
                              Container(widgets=[btn_prev, btn_next], layout="horizontal", labels=False),
                              Container(widgets=[goto, btn_goto], layout="horizontal"),
                              status], scrollable=True)
    # Long code labels would otherwise size the dock to their full text and squeeze the image :
    # combos shrink to ~18 characters (full text in the drop-down) and labels wrap
    def _compact(container) :
        for w in container :
            if isinstance(w, Container) :
                _compact(w)
            elif isinstance(w, ComboBox) :
                w.native.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                w.native.setMinimumContentsLength(18)
            elif isinstance(w, Label) :
                w.native.setWordWrap(True)
    _compact(form)

    # record : landmark -> (row, col) in the sagittal plane (voxels of `image`, axes 1 and 2 of RSA) ;
    # the slice is always the displayed one. moved : landmarks dragged off their parking place.
    # stash : landmarks hidden by a code change, given back if a code needs them again.
    # busy : set while the code itself edits the layers, so their events are ignored
    state = {"case" : None, "t0" : None, "ok" : False, "image" : None, "image_layer" : None,
             "points" : None, "centres" : None, "crosshairs" : None, "record" : {}, "moved" : set(), "stash" : {},
             "busy" : False, "loading" : False,
             # display only, kept from case to case : landmark names (N) and landmarks (H) shown
             "show_names" : True, "show_landmarks" : True}

    # ---- Landmarks ----
    def slice_index() :
        return int(viewer.dims.current_step[0])

    def parking(name) :
        # Column left of the image, one fixed row per landmark of the protocol
        n_rows = state["image"].data.shape[1]
        return (float((lm_order.index(name) + 1) * n_rows / (len(lm_order) + 1)), -8.0)

    def shown() :
        return [n for n in lm_order if n in state["record"]]

    def update_lm_status(*_) :
        if not landmarks :
            return
        if state["points"] is None :
            lm_status.value = ""
            return
        todo = [n for n in shown() if n not in state["moved"]]
        hidden = [w for w, k in (("landmarks (H)", "show_landmarks"), ("names (N)", "show_names")) if not state[k]]
        lm_status.value = (f"slice {slice_index()} - {len(shown())} landmarks<br>"
                           + (f"to drag into place : {', '.join(todo)}" if todo else "all in place")
                           + (f"<br>hidden : {', '.join(hidden)}" if hidden else ""))

    def render() :
        # Layers rebuilt from the record : the only source of truth for the landmarks
        pts, dots = state["points"], state["centres"]
        if pts is None :
            return
        names = shown()
        s = slice_index()
        data = np.array([[s, *state["record"][n]] for n in names], dtype=float).reshape(-1, 3)
        state["busy"] = True
        try :
            pts.selected_data = set()
            pts.data = data
            pts.features = pd.DataFrame({"landmark" : pd.Series(names, dtype=object)})
            pts.refresh()
            draw_marks(data, names)
        finally :
            state["busy"] = False
        update_lm_status()

    # Crosshair of each landmark (as a cross-hair pointer) : 4 coloured arms around a gap and a
    # centre dot, in voxels. The "landmarks" points are only invisible grab handles over them
    # Span 2 x (GAP + ARM) = 2.6 voxels (1.3 mm) ; the grab handle a bit wider, to stay easy to grab
    GAP, ARM, ARM_WIDTH, DOT, HANDLE = 0.4, 0.9, 0.15, 0.4, 3.0
    ARMS = np.array([[0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=float)

    def draw_marks(data, names=None) :
        names = shown() if names is None else names
        colors = [landmarks[n]["color"] for n in names]
        dots, hairs = state["centres"], state["crosshairs"]
        vectors = np.array([[p + GAP * d, ARM * d] for p in data for d in ARMS]).reshape(-1, 2, 3)
        hairs.data = vectors
        dots.data = data
        if names :
            hairs.edge_color = [c for c in colors for _ in ARMS]
            dots.face_color = colors
        hairs.refresh()  # else the canvas can keep the colours of the previous data
        dots.refresh()

    def sync(*_) :
        # Show exactly the landmarks the current codes need
        if state["loading"] or state["points"] is None :
            return
        wanted = set(protocol.required_landmarks({item : item_code(item) for item in fields}))
        for n in list(state["record"]) :
            if n not in wanted :
                state["stash"][n] = (state["record"].pop(n), n in state["moved"])
                state["moved"].discard(n)
        for n in lm_order :
            if n in wanted and n not in state["record"] :
                pos, moved = state["stash"].pop(n, (parking(n), False))
                state["record"][n] = pos
                if moved :
                    state["moved"].add(n)
        render()

    def on_data(event) :
        if state["busy"] :
            return
        action = getattr(event, "action", None)
        action = getattr(action, "value", action)  # napari's ActionType, or its string
        pts = state["points"]
        names = shown()
        if action in ("changing", "changed") and len(pts.data) == len(names) :
            # A drag emits "changing" once, moves with events blocked, then "changed" on release :
            # the record is updated on release (the crosshairs follow live through follow_drag)
            draw_marks(pts.data.copy(), names)
            if action == "changed" :
                for n, p in zip(names, pts.data) :
                    pos = (float(p[1]), float(p[2]))
                    if pos != state["record"][n] :
                        state["record"][n] = pos
                        if pos != parking(n) :
                            state["moved"].add(n)
                update_lm_status()
        elif action in ("added", "removed", "changed") :
            render()  # a point added or removed by napari (key, paste) : undone

    def one_selected(*_) :
        # A box selection would drag several landmarks at once. Checked once the event is over :
        # selecting a point while another is selected passes through a 2-point selection
        def check() :
            pts = state["points"]
            if pts is not None and len(pts.selected_data) > 1 :
                pts.selected_data = set()
        QTimer.singleShot(0, check)

    def select_mode_only(*_) :
        pts = state["points"]
        if pts is not None and pts.mode != "select" :
            pts.mode = "select"

    def follow_drag(layer, event) :
        # napari blocks data events while a point is dragged : the crosshair is moved here
        yield
        while event.type == "mouse_move" :
            if state["centres"] is not None and len(layer.data) == len(shown()) :
                draw_marks(layer.data.copy())
            yield

    def not_middle(drag) :
        # Also no drag while the landmarks are hidden (H) : a landmark cannot be moved unseen
        def wrapped(layer, event) :
            if event.button != MIDDLE and state["show_landmarks"] :
                gen = drag(layer, event)
                if gen is not None :
                    yield from gen
        return wrapped

    def add_landmark_layers() :
        image = state["image"]
        # Grab handles : transparent discs the size of the crosshair, carrying the labels
        pts = viewer.add_points(
            np.empty((0, 3)), ndim=3, name="landmarks", scale=image.spacing, symbol="disc",
            size=HANDLE, face_color="transparent", border_width=0, out_of_slice_display=False,
            features=pd.DataFrame({"landmark" : pd.Series([], dtype=object)}),
            # translation (voxels) : the label above the crosshair, not on it
            text=dict(string="{landmark}", size=7, color="white", anchor="center", translation=(0, -2.2, 0)))
        # Drawn over the handles, never active, so never edited
        hairs = viewer.add_vectors(np.empty((0, 2, 3)), ndim=3, name="crosshairs", scale=image.spacing,
                                   edge_width=ARM_WIDTH, length=1, vector_style="line",
                                   out_of_slice_display=False)
        dots = viewer.add_points(np.empty((0, 3)), ndim=3, name="centres", scale=image.spacing,
                                 size=DOT, border_width=0, out_of_slice_display=False)
        state["points"], state["centres"], state["crosshairs"] = pts, dots, hairs
        apply_display()
        # napari's select mode reacts to every button : a middle drag (pan) would also move or
        # box-select landmarks. Instance copy of the class dict, set before mode = "select" below
        pts._drag_modes = {m : (not_middle(f) if str(m) == "select" else f) for m, f in pts._drag_modes.items()}
        pts.events.data.connect(on_data)
        pts.events.mode.connect(select_mode_only)
        pts.selected_data.events.items_changed.connect(one_selected)
        pts.mouse_drag_callbacks.append(follow_drag)
        viewer.layers.selection.active = pts
        pts.mode = "select"

    def apply_display() :
        # The N / H choices on the layers of the case shown (rebuilt for each case)
        if state["points"] is None :
            return
        state["points"].text.visible = state["show_names"]
        for layer in (state["points"], state["centres"], state["crosshairs"]) :
            layer.visible = state["show_landmarks"]

    def toggle(key) :
        state[key] = not state[key]
        apply_display()
        update_lm_status()

    def set_contrast(*_) :
        if state["image_layer"] is not None and not state["loading"] :
            state["image_layer"].contrast_limits = contrast.value

    # ---- Case display ----
    def open_case(case) :
        viewer.layers.clear()
        state.update(points=None, centres=None, crosshairs=None, image_layer=None, record={}, moved=set(), stash={})
        sub = session.subject(case, modalities)
        for ld in sub.layers("2d") :
            viewer.add_layer(Layer.create(*ld))
        # Axis 0 is left-right (RSA) : start on the middle sagittal slice
        viewer.dims.ndisplay = 2
        viewer.dims.set_current_step(0, viewer.layers[0].data.shape[0] // 2)
        previous = session.get(case)

        state["loading"] = True  # codes set one by one must not hide the landmarks of the others
        try :
            if landmarks :
                # Voxels are those of the first modality (all modalities share the grid)
                state["image"], state["image_layer"] = next(iter(sub.mod.values())), viewer.layers[0]
                lo, hi = (float(v) for v in state["image_layer"].contrast_limits_range)
                contrast.max = max(hi, contrast.max)
                contrast.min, contrast.max = lo, hi
                contrast.value = tuple(float(v) for v in state["image_layer"].contrast_limits)
                saved = session.get_landmarks(case)
                state["record"] = {n : (float(lm["voxel"][1]), float(lm["voxel"][2])) for n, lm in saved.items()}
                state["moved"] = set(saved)
                if saved :
                    viewer.dims.set_current_step(0, int(round(next(iter(saved.values()))["voxel"][0])))
                add_landmark_layers()
            for item, (code_box, quality_box, comment) in fields.items() :
                a = (previous or {}).get(item)
                base, ticked = protocol.split_code(a["code"]) if a else (UNANSWERED, set())
                # A code saved under an older protocol may not exist any more : left to answer again
                code_box.value = base if base in code_box.choices else UNANSWERED
                update_ticks(item)
                for t, cb in tick_boxes[item].items() :
                    cb.value = cb.enabled and t in ticked
                quality_box.value = a["quality"] if a else next(iter(protocol.spec["quality"]))
                comment.value = a["comment"] if a else ""
        finally :
            state["loading"] = False
        sync()  # for a new case : the landmarks needed whatever the codes (the references)
        viewer.reset_view()

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
            state["loading"] = False
            viewer.layers.clear()
            state.update(points=None, centres=None, crosshairs=None, image_layer=None, record={}, moved=set(), stash={})
            update_lm_status()
            for item, (code_box, _, comment) in fields.items() :
                code_box.value, comment.value = UNANSWERED, ""
                update_ticks(item)
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
        answers = {item : dict(code=item_code(item), quality=qb.value, comment=cm.value.strip())
                   for item, (_, qb, cm) in fields.items()}
        blank = [item for item, a in answers.items() if a["code"] == UNANSWERED]
        if blank :
            status.value = f"not saved : no answer for {blank}"
            return
        placed = None
        if landmarks :
            todo = [n for n in shown() if n not in state["moved"]]
            shape = state["image"].data.shape
            outside = [n for n, (r, c) in state["record"].items()
                       if not (0 <= r <= shape[1] - 1 and 0 <= c <= shape[2] - 1)]
            if todo or outside :
                status.value = f"not saved : drag into place {sorted(set(todo) | set(outside))}"
                return
            s = float(slice_index())
            placed = {n : dict(voxel=(s, r, c), mm=state["image"].to_mm((s, r, c)))
                      for n, (r, c) in state["record"].items()}
        try :
            session.save(case, answers, seconds=time.monotonic() - state["t0"], landmarks=placed)
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
    for item, (code_box, _, _) in fields.items() :
        # Connected first : the ticks are settled before sync reads the code
        code_box.changed.connect(lambda *_, item=item : update_ticks(item))
    if landmarks :
        for item, (code_box, _, _) in fields.items() :
            code_box.changed.connect(sync)
            for cb in tick_boxes[item].values() :
                cb.changed.connect(sync)
        def follow_slice(*_) :
            # Only on a real change of sagittal slice : napari also emits current_step when a
            # drag changes the layer's extent, and a render then would undo the drag
            if state["points"] is not None and len(state["points"].data) \
                    and int(round(state["points"].data[0, 0])) != slice_index() :
                render()
        viewer.dims.events.current_step.connect(follow_slice)  # landmarks follow the slice
        contrast.changed.connect(set_contrast)
        btn_reset.changed.connect(viewer.reset_view)
        # Viewer keys, whatever the active layer (napari binds neither key for points or images)
        viewer.bind_key("n", lambda _ : toggle("show_names"), overwrite=True)
        viewer.bind_key("h", lambda _ : toggle("show_landmarks"), overwrite=True)
        # Only the actions of the form : no layer list, no layer controls (add / delete / symbol)
        viewer.window._qt_viewer.dockLayerList.setVisible(False)
        viewer.window._qt_viewer.dockLayerControls.setVisible(False)

    # The scroll area wrapping the form, not `form` : napari would dock form.native, the inner
    # widget, and the dock would keep the full form height as its minimum
    # magicgui fixed the scroll area's minimum width to the uncompacted form (~960 px) when the
    # widgets were inserted, which squeezes the image : let it shrink (it scrolls), start at 500 px
    form.root_native_widget.setMinimumWidth(320)
    dock = viewer.window.add_dock_widget(form.root_native_widget, name="Annotation", area="right")
    viewer.window._qt_window.resizeDocks([dock], [500], Qt.Orientation.Horizontal)
    first = session.next_pending()
    if first is None :
        show(order[0])
        header.value = f"<b>All {len(order)} cases done</b>"
    else :
        show(first)
    return viewer, session, form
