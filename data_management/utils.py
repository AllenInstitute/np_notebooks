from __future__ import annotations

import concurrent.futures
import configparser
import datetime
import functools
import json
import logging
import os
import pathlib
import sys
from typing import Any, Iterable, Literal, Optional, Self

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
assert EPHYS.exists()
UPLOAD = pathlib.Path("//allen/programs/mindscope/workgroups/np-exp/codeocean")
assert UPLOAD.exists()

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


class Config(pydantic.BaseModel):
    folder: str
    ephys_day: Optional[int] = pydantic.Field(
        default=None, description="Day of ephys recording (starting at 1)", gt=0
    )
    perturbation_day: Optional[int] = pydantic.Field(
        default=None,
        description="Day of opto or injection perturbation (starting at 1)",
        gt=0,
    )
    probe_letters_to_skip: Optional[str] = pydantic.Field(
        default="",
        description="Probe letters that weren't inserted or had issues; will be skipped from all further processing (e.g. 'ABC')",
    )
    surface_recording_probe_letters_to_skip: Optional[str] = pydantic.Field(
        default="",
        description="Probe letters that weren't deep insertions; will be skipped from surface channel processing (e.g. 'ABC')",
    )
    is_production: Optional[bool] = pydantic.Field(
        default=True, description="Set to false if this is a test/dev session"
    )
    is_injection_perturbation: Optional[bool] = pydantic.Field(
        default=False, description="Injection perturbation or control session"
    )
    is_opto_perturbation: Optional[bool] = pydantic.Field(
        default=False, description="Optogenetic perturbation or control session"
    )
    is_context_naive: Optional[bool] = pydantic.Field(
        default=False, description="Subject was not trained on stage 3"
    )
    session_type: Literal["ephys", "behavior_with_sync"] = pydantic.Field(
        default="ephys", description="Type of session: ephys or behavior_with_sync"
    )
    project: Literal["DynamicRouting", "TempletonPilotSession"] = pydantic.Field(
        default="DynamicRouting",
        description="Project name: DynamicRouting or TempletonPilotSession",
    )

    @pydantic.field_validator(
        "probe_letters_to_skip",
        "surface_recording_probe_letters_to_skip",
        mode="before",
    )
    def cast_to_upper_case(cls, v):
        return v.upper() if isinstance(v, str) else v

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        return {
            data.pop("session_type"): {
                data.pop("project"): [
                    {
                        f"//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/{data.pop('folder')}": {
                            k: v
                            for k, v in data.items()
                            if v is not None and v != self.model_fields[k].default
                        }
                    }
                ]
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], session_id: str) -> Self:
        """Assumes each session is only in one project/session type"""
        for session_type, project_to_sessions in data.items():
            for project, sessions in project_to_sessions.items():
                if not sessions:
                    continue
                for config in sessions:
                    if isinstance(config, dict):
                        path = next(iter(config.keys()))
                    else:
                        assert isinstance(config, str)
                        path = config
                    if session_id in path:
                        return cls(
                            session_type=session_type,
                            project=project,
                            folder=pathlib.Path(path).name,
                            **(
                                next(iter(config.values()))
                                if isinstance(config, dict)
                                else {}
                            ),
                        )
        return None

    def update_existing_dict(self, existing: dict[str, Any]) -> None:
        data = self.to_dict()
        session_type, project_to_data = next(iter(data.items()))
        project, sessions = next(iter(project_to_data.items()))
        path, config = next(iter(sessions[0].items()))
        for idx, path_to_config in enumerate(
            existing.setdefault(session_type, {}).setdefault(project, [])
        ):
            if self.folder in next(iter(path_to_config.keys())):
                existing[session_type][project].pop(idx)
        existing[session_type][project].append(data[session_type][project][0])


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


class ConfigWidget(ipw.VBox):

    yaml_path = UPLOAD / "new_sessions.yaml"
    placeholders = {
        "probe_letters_to_skip": "e.g. AF",
        "surface_recording_probe_letters_to_skip": "e.g. AF",
        "ephys_day": "starting at 1, or empty if no ephys",
        "perturbation_day": "starting at 1, or empty if no perturbation",
        "session_type": "ephys or behavior_with_sync",
    }

    def __init__(self, data: dict[str, Any], **vbox_kwargs):
        self.session_folder = data["folder"]
        self.config = Config(**data)
        self.console = ipw.Output()
        self.update_with_previous_data()

        self.text_entry_boxes = {
            name: ipw.Text(
                description=name,
                placeholder=self.placeholders.get(name, ""),
                tooltip=field.description or name,
                continuous_update=True,
                layout=ipw.Layout(width="500px", description_width="400px"),
                value=(
                    str(getattr(self.config, name))
                    if getattr(self.config, name) is not None
                    else ""
                ),
                style={"description_width": "initial"},
            )
            for name, field in self.config.model_fields.items()
        }
        self.text_entry_grid = ipw.GridBox(
            list(self.text_entry_boxes.values()),
        )
        self.save_button = ipw.Button(
            description="Save",
            button_style="warning",
            layout=ipw.Layout(width="30%"),
            tooltip="Save yaml config for this session",
        )

        def on_save_click(widget):
            self.save()
            widget.button_style = "success"

        self.save_button.on_click(on_save_click)

        super().__init__(
            [ipw.HBox([self.text_entry_grid]), self.save_button, self.console],
            **vbox_kwargs,
        )

    def update_with_previous_data(self) -> None:
        if self.yaml_path.exists():
            with self.console:
                existing = yaml.safe_load(self.yaml_path.read_text()) or {}
                if not existing:
                    return None
                e = Config.from_dict(existing, self.session_folder)
                if e is None:
                    return None
                self.config = e
                print(f"Loaded existing data for {self.session_folder}")

    def update_from_text_boxes(self) -> None:
        with self.console:
            self.config = Config(
                **{
                    name: box.value if box.value != "" else None
                    for name, box in self.text_entry_boxes.items()
                },
            )

    def save(self) -> None:
        with self.console:
            self.update_from_text_boxes()
            if self.yaml_path.exists():
                print(f"Updating {self.yaml_path}")
                existing = yaml.safe_load(self.yaml_path.read_text()) or {}
            else:
                print(f"Creating {self.yaml_path}")
                self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
                existing = {}
            self.config.update_existing_dict(existing)
            self.yaml_path.write_text(yaml.dump(existing))
            print("Done")


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
    for folder in session_folders:
        if "surface_channel" in folder:
            print(
                "Skipping surface channel folder: metadata is same as main session folder"
            )
            continue  # metadata for surface channel is same as main session folder
        row = get_folder_df(ttl_hash=aind_session.get_ttl_hash(600)).filter(
            pl.col("folder") == folder
        )
        print("")
        IPython.display.display(
            ConfigWidget(
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
        )


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
        row["ephys"] = bool(next((EPHYS / s).rglob('settings*.xml'), None))
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
    for row in get_folder_df(ttl_hash=aind_session.get_ttl_hash(600)).iter_rows(named=True):
        if not row['ephys']:
            continue
        for config_yaml in config_yamls:
            config = Config.from_dict(
                yaml.safe_load(config_yaml.read_text()) or {}, row['folder']
            )
            if config is not None:
                break
        else:
            sessions.append(row['folder'])
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
        value=df.sort_values(by='subject'),
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
