"""
Command line launcher, installed as `cerebellum-viz`.

    cerebellum-viz view <dataset_root> <sid> [-m t1w zscore] [--ses ses-01] [--3d]
    cerebellum-viz annotate <project_root> <rater> [--data-root <dataset_root>]
    cerebellum-viz status <project_root>

--data-root : where the dataset is on this computer, when it differs from the path stored in the
project (project.json). An unreadable data root is reported as an error before napari opens.
"""

import argparse


def main(argv=None) :
    parser = argparse.ArgumentParser(prog="cerebellum-viz")
    sub = parser.add_subparsers(dest="command", required=True)

    p_view = sub.add_parser("view", help="display one subject in napari")
    p_view.add_argument("root", help="dataset root")
    p_view.add_argument("sid")
    p_view.add_argument("-m", "--modalities", nargs="+", default=None,
                        help="modality keys (default : every key of the layout)")
    p_view.add_argument("--ses", default=None, help="session (uses the session layout)")
    p_view.add_argument("--3d", dest="three_d", action="store_true", help="3D surfaces instead of slices")

    p_ann = sub.add_parser("annotate", help="open a rater's annotation session")
    p_ann.add_argument("project", help="annotation project root (holds project.json)")
    p_ann.add_argument("rater")
    p_ann.add_argument("--data-root", default=None,
                       help="dataset root on this computer (default : the one stored in project.json)")

    p_status = sub.add_parser("status", help="print an annotation project's status")
    p_status.add_argument("project")

    args = parser.parse_args(argv)

    if args.command == "status" :
        from .annotation import AnnotationProject
        AnnotationProject.load(args.project).status()
        return

    import napari
    if args.command == "view" :
        from .viewer import view
        view(args.root, args.sid, args.modalities, ses=args.ses,
             layout="ses" if args.ses else "default", mode="3d" if args.three_d else "2d")
    else :
        from .annotation import DataRootError
        from .viewer import annotate
        try :
            annotate(args.project, args.rater, data_root=args.data_root)
        except DataRootError as e :
            parser.exit(1, f"cerebellum-viz: error: {e}\n")
    napari.run()


if __name__ == "__main__" :
    main()
