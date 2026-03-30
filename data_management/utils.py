from __future__ import annotations

import concurrent.futures
import configparser
import datetime
import functools
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
from typing import Any, Iterable, Literal, Optional, Union, get_args, get_origin

import aind_session
import IPython
import IPython.display
import ipywidgets as ipw
import np_config
import np_tools
import npc_lims
import npc_session
import pandas as pd
import panel as pn
import polars as pl
import pydantic
import yaml

pl.Config.set_tbl_rows(-1)

# suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

os.environ["CODE_OCEAN_API_TOKEN"] = np_config.from_zk(
    "/projects/np_codeocean/codeocean"
)["credentials"]["token"]


# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# pn.config.theme = "dark"
pn.config.notifications = True
pn.extension("tabulator")

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

logging.getLogger("aind_session.utils.docdb_utils").disabled = True
logging.getLogger("aind_session.utils.codeocean_utils").disabled = True
logging.getLogger("aind_session.session").disabled = True
logging.getLogger("aind_session.utils").disabled = True

EPHYS = pathlib.Path(
    "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot"
)
UPLOAD = pathlib.Path("//allen/programs/mindscope/workgroups/np-exp/codeocean")

executor = concurrent.futures.ThreadPoolExecutor()
expected_suffixes = {".h5": "sync", ".hdf5": "stim", ".mp4": "video"}


def get_aws_files() -> dict[Literal["config", "credentials"], pathlib.Path]:
    return {
        "config": pathlib.Path("~").expanduser() / ".aws" / "config",
        "credentials": pathlib.Path("~").expanduser() / ".aws" / "credentials",
    }


def get_codeocean_files() -> dict[Literal["credentials"], pathlib.Path]:
    return {
        "credentials": pathlib.Path("~").expanduser()
        / ".codeocean"
        / "credentials.json",
    }


def verify_ini_config(
    path: pathlib.Path, contents: dict, profile: str = "default"
) -> None:
    config = configparser.ConfigParser()
    if path.exists():
        config.read(path)
    if not all(k in config[profile] for k in contents):
        raise ValueError(
            f"Profile {profile} in {path} exists but is missing some keys required for codeocean or s3 access."
        )


def write_or_verify_ini_config(
    path: pathlib.Path, contents: dict, profile: str = "default"
) -> None:
    config = configparser.ConfigParser()
    if path.exists():
        config.read(path)
        try:
            verify_ini_config(path, contents, profile)
        except ValueError:
            pass
        else:
            return
    config[profile] = contents
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("w") as f:
        config.write(f)
    verify_ini_config(path, contents, profile)


def verify_json_config(path: pathlib.Path, contents: dict) -> None:
    config = json.loads(path.read_text())
    if not all(k in config for k in contents):
        raise ValueError(
            f"{path} exists but is missing some keys required for codeocean or s3 access."
        )


def write_or_verify_json_config(path: pathlib.Path, contents: dict) -> None:
    if path.exists():
        try:
            verify_json_config(path, contents)
        except ValueError:
            contents = np_config.merge(json.loads(path.read_text()), contents)
        else:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    path.write_text(json.dumps(contents, indent=4))


def ensure_credentials() -> None:
    for file, contents in (
        (get_aws_files()["config"], get_aws_config()),
        (get_aws_files()["credentials"], get_aws_credentials()),
    ):
        write_or_verify_ini_config(file, contents, profile="default")

    for file, contents in (
        (get_codeocean_files()["credentials"], get_codeocean_config()),
    ):
        write_or_verify_json_config(file, contents)


@functools.cache
def get_aws_config() -> (
    dict[Literal["aws_access_key_id", "aws_secret_access_key"], str]
):
    """Config for connecting to AWS/S3 via awscli/boto3"""
    return np_config.fetch("/projects/np_codeocean/aws")["config"]


@functools.cache
def get_aws_credentials() -> dict[Literal["domain", "token"], str]:
    """Config for connecting to AWS/S3 via awscli/boto3"""
    return np_config.fetch("/projects/np_codeocean/aws")["credentials"]


@functools.cache
def get_codeocean_config() -> dict[Literal["region"], str]:
    """Config for connecting to CodeOcean via http API"""
    return np_config.fetch("/projects/np_codeocean/codeocean")["credentials"]



# session config ----------------------------------------------------------------------- #

_DR = "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot"
_TEMPLETON = "//allen/programs/mindscope/workgroups/templeton/TTOC/pilot recordings"
PROJECT_PATHS: dict[str, str] = {
    "DynamicRouting": _DR,
    "TempletonPilotSession": _TEMPLETON,
}

EPHYS = pathlib.Path(PROJECT_PATHS["DynamicRouting"])


def get_session_config_path(
    folder: str, project: str = "DynamicRouting"
) -> pathlib.Path:
    return pathlib.Path(PROJECT_PATHS[project]) / folder / "session_config.json"


class Config(pydantic.BaseModel):
    folder: str
    project: Literal["DynamicRouting", "TempletonPilotSession"] = pydantic.Field(
        default="DynamicRouting",
        description="Project name: DynamicRouting or TempletonPilotSession",
    )
    session_type: Literal["ephys", "behavior_with_sync"] = pydantic.Field(
        default="ephys", description="Type of session: ephys or behavior_with_sync"
    )
    ephys_day: Optional[int] = pydantic.Field(
        default=None, description="Day of ephys recording (starting at 1)", gt=0
    )
    perturbation_day: Optional[int] = pydantic.Field(
        default=None,
        description="Day of opto or injection perturbation (starting at 1)",
        gt=0,
    )
    is_production: bool = pydantic.Field(
        default=True,
        description="Production quality data; experimental variants are ok (False: dev testing, training operators)",
    )
    is_split_recording: bool = pydantic.Field(
        default=False,
        description="Split recording session: will not be uploaded yet (recordings to be concatenated later)",
    )
    is_context_naive: bool = pydantic.Field(
        default=False,
        description="Subject was not trained on stage 3 before first experiment",
    )
    is_injection_perturbation: bool = pydantic.Field(
        default=False, description="Injection perturbation or control session"
    )
    is_opto_perturbation: bool = pydantic.Field(
        default=False, description="Optogenetic perturbation or control session"
    )
    is_deep_insertion: bool = pydantic.Field(
        default=False,
        description="At least one probe has a surface channel recording",
    )
    probe_letters_to_skip: Optional[str] = pydantic.Field(
        default="",
        description="Probe letters to skip from upload/processing (e.g. 'ABC', [A-F], max 6 chars). Not necessary to list probes that were disabled in Open Ephys",
    )
    surface_recording_probe_letters_to_skip: Optional[str] = pydantic.Field(
        default="",
        description="Probe letters to skip from surface channel processing (e.g. 'ABC', [A-F], max 6 chars). Not necessary to list probes that were disabled in Open Ephys",
    )

    @pydantic.field_validator(
        "probe_letters_to_skip",
        "surface_recording_probe_letters_to_skip",
        mode="before",
    )
    def cast_to_upper_case(cls, v):
        return v.upper() if isinstance(v, str) else v

    @pydantic.field_validator(
        "probe_letters_to_skip",
        "surface_recording_probe_letters_to_skip",
        mode="after",
    )
    def validate_probe_letters(cls, v):
        if v and not re.fullmatch(r"[A-F]{0,6}", v):
            raise ValueError("Probe letters must be A-F only, up to 6 characters")
        return v

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        session_type = data.pop("session_type")
        project = data.pop("project")
        folder = data.pop("folder")
        return {
            session_type: {
                project: [
                    {
                        f"{PROJECT_PATHS[project]}/{folder}": {
                            "ephys_day": self.ephys_day,
                            "session_kwargs": {
                                k: v
                                for k, v in data.items()
                                if v is not None
                                and v != self.model_fields[k].default
                                and k not in ("ephys_day", "perturbation_day")
                            },
                        }
                    }
                ]
            }
        }

    def to_yaml_text_snippet(self) -> str:
        d = self.to_dict()
        indent = " " * 4
        session_dir_parent = PROJECT_PATHS[self.project] + "/"
        s = f"\n{indent}- {session_dir_parent}{self.folder}:"
        for attr in (
            "ephys_day",
            "perturbation_day",
        ):
            if value := getattr(self, attr, None):
                s = s + "\n" + indent * 2 + f"{attr}: {value}"
        session_kwargs = next(
            iter(next(iter(d[self.session_type][self.project])).values())
        )["session_kwargs"]
        if session_kwargs:
            s = s + "\n" + indent * 2 + "session_kwargs:"
            for k, v in session_kwargs.items():
                s = s + "\n" + indent * 3 + f"{k}: {v}"
        if s.endswith(":"):
            s = s[:-1]
        s = s.replace("\n\n", "\n")
        return (
            s
            + "\n"
            + (
                indent
                if (self.project == "DynamicRouting" and self.session_type == "ephys")
                else ""
            )
        )


class SessionConfigRow:
    """Widget row for a single session's configuration."""

    placeholders = {
        "probe_letters_to_skip": "e.g. AF",
        "surface_recording_probe_letters_to_skip": "e.g. AF",
        "ephys_day": "starting at 1, or empty if no ephys",
        "perturbation_day": "starting at 1, or empty if no perturbation",
    }

    @staticmethod
    def _is_bool_field(field) -> bool:
        """Check if a pydantic field is a boolean type."""
        if field.annotation is bool:
            return True
        if get_origin(field.annotation) is Union:
            return bool in get_args(field.annotation)
        return False

    @staticmethod
    def _is_literal_field(field) -> bool:
        """Check if a pydantic field is a Literal type."""
        return get_origin(field.annotation) is Literal

    @staticmethod
    def _make_description_label(description: str | None) -> ipw.HTML:
        """Create a small italic caption from a field description."""
        text = description or ""
        return ipw.HTML(
            value=f'<span style="color: #888; font-size: 0.85em; font-style: italic;">{text}</span>',
            layout=ipw.Layout(margin="0 0 4px 160px"),
        )

    def __init__(self, data: dict[str, Any]):
        self.session_folder = data["folder"]
        config_path = get_session_config_path(self.session_folder)
        if config_path.exists():
            saved = json.loads(config_path.read_text())
            data = {**data, **saved}
        self.config = Config(**data)
        self.widgets = (
            {}
        )  # field_name -> (input_widget, description_label) or HTML for folder

        for name, field in self.config.model_fields.items():
            if name == "folder":
                self.widgets[name] = (
                    ipw.HTML(
                        value=f"<b>{getattr(self.config, name)}</b>",
                        layout=ipw.Layout(width="400px"),
                    ),
                    None,
                )
            elif self._is_bool_field(field):
                current_value = getattr(self.config, name)
                if current_value is None:
                    current_value = True
                self.widgets[name] = (
                    ipw.Dropdown(
                        description=name,
                        options=[("True", True), ("False", False)],
                        value=current_value,
                        tooltip=field.description or name,
                        layout=ipw.Layout(width="500px"),
                        style={"description_width": "initial"},
                    ),
                    self._make_description_label(field.description),
                )
            elif self._is_literal_field(field):
                options = list(getattr(field.annotation, "__args__", []))
                current_value = getattr(self.config, name)
                self.widgets[name] = (
                    ipw.Dropdown(
                        description=name,
                        options=options,
                        value=current_value,
                        tooltip=field.description or name,
                        layout=ipw.Layout(width="500px"),
                        style={"description_width": "initial"},
                    ),
                    self._make_description_label(field.description),
                )
            else:
                self.widgets[name] = (
                    ipw.Text(
                        description=name,
                        placeholder=self.placeholders.get(name, ""),
                        tooltip=field.description or name,
                        continuous_update=True,
                        layout=ipw.Layout(width="500px"),
                        value=(
                            str(getattr(self.config, name))
                            if getattr(self.config, name) is not None
                            else ""
                        ),
                        style={"description_width": "initial"},
                    ),
                    self._make_description_label(field.description),
                )

        self._setup_autosave()

    def get_config(self) -> Config:
        """Get Config object from current widget values."""
        data = {}
        for name, (widget, _label) in self.widgets.items():
            if isinstance(widget, ipw.HTML):
                data[name] = self.session_folder
            elif isinstance(widget, ipw.Dropdown):
                data[name] = widget.value
            else:
                data[name] = widget.value if widget.value != "" else None
        return Config(**data)

    def save_to_session_folder(self) -> pathlib.Path:
        """Save current config as JSON to the session folder."""
        config = self.get_config()
        path = get_session_config_path(self.session_folder)
        path.write_text(json.dumps(config.model_dump(), indent=2))
        return path

    def _setup_autosave(self) -> None:
        """Observe all input widgets and autosave on any change."""
        self.status_label = ipw.HTML(value="")

        def _autosave(change):
            try:
                self.save_to_session_folder()
                self.status_label.value = ""
            except pydantic.ValidationError as e:
                msgs = "; ".join(err["msg"] for err in e.errors())
                self.status_label.value = f'<span style="color: red;">{msgs}</span>'
            except FileNotFoundError:
                self.status_label.value = '<span style="color: orange;">Session folder not found on network drive — config not saved</span>'

        for widget, _label in self.widgets.values():
            if not isinstance(widget, ipw.HTML):
                widget.observe(_autosave, names="value")

    def iter_display_widgets(self):
        """Yield flat sequence of (input_widget, description_label) for display."""
        for widget, label in self.widgets.values():
            yield widget
            if label is not None:
                yield label
        yield self.status_label


class CombinedConfigWidget(ipw.VBox):
    """Combined widget for all sessions with a single save button."""

    def __init__(self, session_data_list: list[dict[str, Any]], **vbox_kwargs):
        self.session_rows = [SessionConfigRow(data) for data in session_data_list]
        self.console = ipw.Output()

        header = ipw.HTML(value="<h3>Session metadata (auto-saves)</h3>")

        all_widgets = []
        for row in self.session_rows:
            all_widgets.append(ipw.HTML(value="<hr>"))
            all_widgets.extend(row.iter_display_widgets())

        widget_grid = ipw.VBox(all_widgets)

        self.save_to_npc_lims = ipw.Button(
            description="[upload only] Save & Push to GitHub (double-check!)",
            button_style="warning",
            layout=ipw.Layout(width="30%"),
            tooltip="Save yaml config for all sessions and push to GitHub",
        )

        def on_save_to_npc_lims_click(widget):
            widget.disabled = True
            with self.console:
                for row in self.session_rows:
                    path = row.save_to_session_folder()
                    print(f"Saved {path}")
            self.save_and_push()
            widget.button_style = "success"
            widget.disabled = False

        self.save_to_npc_lims.on_click(on_save_to_npc_lims_click)

        bottom = [] if os.environ.get("AIBS_RIG_ID") else [self.save_to_npc_lims]
        super().__init__(
            [header, widget_grid, *bottom, self.console],
            **vbox_kwargs,
        )

    def get_existing_sessions(self, yml_path: pathlib.Path) -> set[str]:
        """Get set of existing session paths from yaml file."""
        if not yml_path.exists():
            return set()

        existing = yaml.safe_load(yml_path.read_text()) or {}
        session_paths = set()

        for session_type, project_data in existing.items():
            if not isinstance(project_data, dict):
                continue
            for project, sessions in project_data.items():
                if not isinstance(sessions, list):
                    continue
                for session in sessions:
                    if isinstance(session, dict):
                        for path in session.keys():
                            session_paths.add(path)

        return session_paths

    def save_and_push(self) -> None:
        """Save all configs to yaml and push to GitHub."""
        with self.console:
            try:
                root = pathlib.Path().resolve().parent.parent
                repo_path = root / "npc_lims"
                yml_path = repo_path / "tracked_sessions.yaml"

                if not yml_path.exists():
                    raise FileNotFoundError(
                        f"git clone npc_lims into {root} before trying to update tracked_sessions.yaml"
                    )

                existing_sessions = self.get_existing_sessions(yml_path)
                new_configs = [row.get_config() for row in self.session_rows]

                duplicates = [
                    config.folder
                    for config in new_configs
                    if f"{PROJECT_PATHS[config.project]}/{config.folder}"
                    in existing_sessions
                ]

                if duplicates:
                    raise ValueError(
                        f"The following sessions are already in tracked_sessions.yaml: {', '.join(duplicates)}. "
                        f"To modify existing sessions, make changes directly in GitHub."
                    )

                txt = yml_path.read_text()

                for config in new_configs:
                    if config.session_type == "ephys":
                        ephys_stop = txt.find("behavior_with_sync:")
                        if config.project == "TempletonPilotSession":
                            stop = ephys_stop
                        else:
                            stop = txt[:ephys_stop].find("TempletonPilotSession:")
                    else:
                        assert config.session_type == "behavior_with_sync"
                        stop = len(txt)

                    txt = (
                        txt[:stop]
                        + "\n"
                        + config.to_yaml_text_snippet()
                        + "\n"
                        + ("  " if config.project == "DynamicRouting" else "")
                        + (txt[stop:] if stop else "\n")
                    )

                print("Updating tracked_sessions.yaml...")
                yml_path.write_text(txt)

                print("Committing changes...")
                subprocess.run(
                    ["git", "add", "tracked_sessions.yaml"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )

                commit_msg = f"Auto add metadata for {len(new_configs)} session(s)"
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )

                print("Pushing to GitHub...")
                subprocess.run(
                    ["git", "push"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                )

                print(
                    f"✓ Successfully saved and pushed metadata for {len(new_configs)} session(s)!"
                )

            except subprocess.CalledProcessError as e:
                print(f"Git error: {e}")
                if e.stderr:
                    print(f"Error output: {e.stderr.decode()}")
                raise
            except Exception as e:
                print(f"Error: {e}")
                raise

class UploadWidget(ipw.VBox):

    def __init__(self, folder: str, **vbox_kwargs):
        self.folder = folder
        self.console = ipw.Output()
        self.upload_button = ipw.Button(
            description=f"Upload {self.folder}",
            button_style="warning",
            layout=ipw.Layout(width="50%"),
            tooltip="Upload files to S3",
        )
        self.force_toggle = ipw.Checkbox(
            description="Force overwrite existing files",
            value=False,
        )

        def on_upload_click(widget):
            widget.disabled = True
            self.upload()
            widget.button_style = "success"

        self.upload_button.on_click(on_upload_click)

        super().__init__(
            [self.upload_button, self.force_toggle, self.console],
            **vbox_kwargs,
        )

    def upload(self) -> None:
        with self.console:
            (pathlib.Path.cwd() / "logs").mkdir(exist_ok=True, parents=True)
            import np_codeocean.scripts.upload_dynamic_routing_ecephys as upload_dr_ecephys

            upload_dr_ecephys.write_metadata_and_upload(
                self.folder, force=self.force_toggle.value
            )
            # executable = str(pathlib.Path(sys.executable).with_name("upload_dr_ecephys.exe"))
            # subprocess.Popen(
            #     args=[self.folder] + (["--force"] if self.force_toggle.value else []),
            #     executable=executable,
            #     check=True,
            #     # capture_output=True,
            #     stderr=subprocess.PIPE,
            # )
            # print(f"Submitted. Check progress here: http://aind-data-transfer-service/jobs")


# def toggle_tracebacks() -> Generator[None, None, None]:
#     if ipython := IPython.get_ipython():
#         show_traceback = ipython.showtraceback

#         def hide_traceback(exc_tuple=None, filename=None, tb_offset=None,
#                         exception_only=False, running_compiled_code=False):
#             etype, value, tb = sys.exc_info()
#             return ipython._showtraceback(etype, value, ipython.InteractiveTB.get_exception_only(etype, value))

#         hidden = True
#         while True:
#             ipython.showtraceback = hide_traceback if hidden else show_traceback
#             hidden = yield
#     else:
#         raise RuntimeError("Not in IPython")
# toggle_tb = toggle_tracebacks()
# toggle_tb.send(None)
# def show_tracebacks():
#     toggle_tb.send(False)
# def hide_tracebacks():
#     toggle_tb.send(True)


def get_folder_contents_df(folder_name: str) -> pl.DataFrame | None:
    folder_contents = []
    folder_path = EPHYS / folder_name

    def add_contents(path: pathlib.Path):

        folder_contents.append(
            {
                "folder": folder_name,
                "type": (
                    expected_suffixes.get(path.suffix)
                    if "Record Node" not in path.name
                    else "ephys"
                ),
                "name": path.stem,
                "suffix": path.suffix or "",
                "size MB": (s := np_tools.size(path)) // 1024**2,
                "size GB": round(s / 1024**3, 1),
            }
        )

    for path in folder_path.glob("*"):
        if path.is_dir() or path.suffix in expected_suffixes:
            if path.name == folder_name:  # ephys folder
                for subpath in path.glob("*"):
                    add_contents(subpath)
            else:
                add_contents(path)

    if not folder_contents:
        return None
    return pl.DataFrame(folder_contents).sort("suffix", "name")


def validate_folder_contents(folder_names: Iterable[str]) -> None:
    if not folder_names:
        raise ValueError("Select at least one folder")

    for folder_name in folder_names:
        print(folder_name)
        folder_contents_df = get_folder_contents_df(folder_name)
        folder_path = EPHYS / folder_name
        is_surface_channel = "surface_channel" in folder_name
        missing = []
        if folder_contents_df is None:
            # add everything
            missing.append("ephys")
            if not is_surface_channel:
                missing.extend(expected_suffixes.values())
        else:
            if not (
                folder_contents_df.get_column("name")
                .str.contains_any(["Record Node"])
                .any()
            ):
                missing.append("ephys")
            if not is_surface_channel:
                for suffix in expected_suffixes:
                    if suffix not in folder_contents_df["suffix"]:
                        missing.append(expected_suffixes[suffix])
        if missing:
            print(f"\tmissing data: {missing}")
        else:
            print("\tappears ready for upload")
        print(f"\t{folder_path}\n")


def display_config_widgets(session_folders: Iterable[str]) -> None:
    """Display combined config widget for all sessions."""
    # Filter out surface channel folders and prepare data
    session_data_list = []
    folder_df = get_folder_df(ttl_hash=aind_session.get_ttl_hash(600))

    for folder in sorted(session_folders):
        if "surface_channel" in folder:
            print(f"Skipping {folder}: metadata is same as main session folder")
            continue

        row = folder_df.filter(pl.col("folder") == folder)
        session_data_list.append(
            dict(
                folder=folder,
                ephys_day=row["ephys day"][0],
                probe_letters_to_skip="",
                surface_recording_probe_letters_to_skip="",
                is_production=True,
                is_injection_perturbation=False,
                is_opto_perturbation=False,
                session_type="ephys" if row["ephys"][0] else "behavior_with_sync",
                project="DynamicRouting",
            )
        )

    if session_data_list:
        IPython.display.display(CombinedConfigWidget(session_data_list))


def display_upload_widgets(session_folders: Iterable[str]) -> None:
    for folder in session_folders:
        print("")
        IPython.display.display(UploadWidget(folder))


@functools.cache
def get_folder_df(ttl_hash: int | None = None):
    del ttl_hash

    folders = [
        p.name for p in EPHYS.glob("DRpilot*") if p.is_dir() and "366122" not in p.name
    ]
    columns = (
        "subject",
        "ephys",
        "date",
        "folder",
        "created",
        "aind ID",
        "uploaded",
    )
    records = []
    logger.info(f"Submitting {len(folders)} jobs to threadpool")

    def get_row(s: str):
        logger.info(f"Fetching info for {s}")
        row = dict.fromkeys(columns, None)
        row["folder"] = s
        row["ephys"] = bool(next((EPHYS / s).rglob("settings*.xml"), None))
        row["date"] = npc_session.extract_isoformat_date(s)
        row["subject"] = str(npc_session.extract_subject(s))
        upload = UPLOAD / s / "upload.csv"
        row["created"] = upload.exists()
        if row["created"]:
            df = pl.read_csv(upload)
            if any("acq-datetime" in c for c in df.columns):
                for c in df.columns:
                    if "acq-datetime" in c:
                        dt = df[c].drop_nulls()[0]
                        break
            elif r"acq-datetime\r" in df.columns:
                dt = df[r"acq-datetime\r"].drop_nulls()[0]
            elif "acq-date" in df.columns:
                dt = (
                    f"{df['acq-date'].drop_nulls()[0]}_{df['acq-time'].drop_nulls()[0]}"
                )
            else:
                raise ValueError(f"no datetime column found in {upload}")
            if not dt:
                import pdb

                pdb.set_trace()
            dt = dt.replace(":", "-").replace(" ", "_")
            row["aind ID"] = npc_session.AINDSessionRecord(
                f"ecephys_{row['subject']}_{dt}"
            ).id
            row["uploaded"] = aind_session.Session(row["aind ID"]).is_uploaded
        return row

    for row in executor.map(get_row, folders):
        records.append(row)
    if not records:
        df = pl.DataFrame(columns)
    else:
        df = pl.DataFrame(records)

    df = (
        df.sort("date", descending=False)
        # .filter(pl.col("ephys"))
        .with_columns(
            pl.when(pl.col("ephys").eq(True))
            .then(pl.col("date").cum_count().over(pl.col("ephys"), pl.col("subject")))
            .otherwise(pl.lit(None))
            .cast(str)
            .alias("ephys day")
        ).sort("date", "subject", descending=True)
    )
    return df


def get_folders_with_no_metadata() -> tuple[str, ...]:
    sessions = []
    config_yamls = [
        c
        for c in (
            npc_lims.tracked_sessions._TRACKED_SESSIONS_FILE,
            UPLOAD / "new_sessions.yaml",
        )
        if c.exists()
    ]
    for row in get_folder_df(ttl_hash=aind_session.get_ttl_hash(600)).iter_rows(
        named=True
    ):
        if not row["ephys"]:
            continue
        for config_yaml in config_yamls:
            config = Config.from_dict(
                yaml.safe_load(config_yaml.read_text()) or {}, row["folder"]
            )
            if config is not None:
                break
        else:
            sessions.append(row["folder"])
    return tuple(sorted(sessions, reverse=True))


def get_folder_table(
    ephys_only: bool = False,
    unstarted_only: bool = False,
) -> pn.widgets.Tabulator:
    """Get sessions for a specific subject and date range."""
    # yield pn.indicators.LoadingSpinner(
    #     value=True, size=20, name="Fetching folders..."
    # )
    df = get_folder_df(ttl_hash=aind_session.get_ttl_hash(600)).to_pandas()
    if ephys_only:
        df = df[df["ephys"]]
    if unstarted_only:
        df = df[~df["created"]]
    column_filters = {
        "subject": {"type": "input", "func": "like", "placeholder": "like x"},
        "folder": {"type": "input", "func": "like", "placeholder": "like x"},
        # "ephys": {
        #     "type": "tickCross",
        #     "tristate": True,
        #     "indeterminateValue": None,
        #     "defaultValue": True,
        # },
        # "created": {"type": "tickCross", "tristate": True, "indeterminateValue": None},
        # "uploaded": {"type": "tickCross", "tristate": True, "indeterminateValue": None},
    }

    # Custom formatter to highlight cells with False in the success column
    def color_negative_red(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for negative
        bools, black otherwise.
        """
        color = (
            "red" if not val else ("white" if pn.config.theme == "dark" else "black")
        )
        return "color: %s" % color

    stylesheet = """
    .tabulator-cell {
        font-size: 12px;
    }
    """
    table = pn.widgets.Tabulator(
        hidden_columns=["date"] + (["ephys"] if ephys_only else []),
        # groupby=["subject"],
        page_size=15,
        value=df.sort_values(by="subject"),
        selectable="checkbox-single",
        # disabled=True,
        show_index=False,
        sizing_mode="stretch_width",
        # row_content=content_fn,
        widths={"folder": "20%", "aind ID": "20%"},
        embed_content=False,
        stylesheets=[stylesheet],
        formatters={
            "bool": {"type": "tickCross"},  # not working
        },
        header_filters=column_filters,
    )
    table.style.map(color_negative_red)
    return table


def is_split_recording(folder: str) -> bool:
    """Check if session_config.json indicates a split recording."""
    path = EPHYS / folder / "session_config.json"
    if not path.exists():
        logger.warning(f"No session_config.json found for {folder}")
        return False
    config = json.loads(path.read_text())
    return bool(config.get("is_split_recording"))


def create_bat_file(
    folder_names: Iterable[str],
    path: pathlib.Path = pathlib.Path("~/Desktop/upload_s3.bat").expanduser(),
) -> None:
    txt = f"@REM {datetime.datetime.now().isoformat()}\n"
    txt += f"CD {pathlib.Path.cwd()}\n\n"
    for folder_name in folder_names:
        txt += f"CALL {pathlib.Path(sys.executable).with_name('upload_dr_ecephys.exe')} {folder_name}\n"
    if path.exists():
        existing = path.read_text()
        txt += "\n"
        txt += "".join(
            f"{'@REM ' if not line.startswith('@REM ') else ''}{line}\n"
            for line in existing.splitlines()
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt)
