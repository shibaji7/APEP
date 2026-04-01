import matlab
import matlab.engine
import numpy as np
import pandas as pd
from loguru import logger


def get_matlab_pynasonde_lib():
    import importlib.util
    import os
    import pathlib

    pynasonde_spec = importlib.util.find_spec("pynasonde")
    if pynasonde_spec is None:
        raise ImportError("pynasonde module not found")
    pynasonde_path = pathlib.Path(pynasonde_spec.origin).parent
    lib_path = pynasonde_path / "matlab_lib"
    if lib_path.exists():
        logger.info(f"Matlab library path found: {lib_path}")
        return str(lib_path)
    else:
        raise FileNotFoundError(f"Matlab library path not found: {lib_path}")


class CreateFig:

    def __init__(self, fig_path: str = "figures/", lib_path: str = None):
        self.fig_path = fig_path
        self.eng = matlab.engine.start_matlab()
        env_path = get_matlab_pynasonde_lib() if lib_path is None else lib_path
        self.eng.addpath(self.eng.genpath(env_path), nargout=0)
        logger.info("Matlab engine started and library path added.")
        return

    def close(self):
        logger.info("Closing Matlab engine.")
        self.eng.quit()
        return

    def generate_scaled_TS_figure(
        self,
        data_dicts: list[dict],
        fig_file_name: str,
        fig_title: str = "",
        fontsize: int = 16,
        fig_shape: tuple = (8, 2),
    ):
        self.eng.eval(
            f"""
            sp = SaoSummaryPlots("{fig_title}", {len(data_dicts)}, 1, {fontsize}, [{fig_shape[0]} {fig_shape[1]}*{len(data_dicts)}]);
            """,
            nargout=0,
        )
        for i, data_dict in enumerate(data_dicts):
            (
                df,
                background,
                datetime_key,
                xlim,
                right_yparams,
                left_yparams,
                left_param_labels,
                right_param_labels,
                color_direction,
                title_txt,
                vlines,
                vline_styles,
            ) = (
                data_dict["dataset"],
                data_dict.get("background", None),
                data_dict.get("datetime_key", "datetime"),
                data_dict.get("xlim", []),
                data_dict.get("right_yparams", ["hmF2", "hmE"]),
                data_dict.get("left_yparams", ["foF2", "foE"]),
                data_dict.get("left_param_labels", ["hmF_2", "hmE"]),
                data_dict.get("right_param_labels", ["foF_2", "foE"]),
                data_dict.get("color_direction", "dark2light"),
                data_dict.get("title_txt", f"({chr(65+i)})"),
                data_dict.get("vlines", []),
                data_dict.get("vline_styles", []),
            )

            num_cols, str_cols = (
                df.select_dtypes(include=["number"]).columns,
                df.columns.difference(df.select_dtypes(include=["number"]).columns),
            )

            # numeric as matlab.double
            num_mat, dt_str, str_mat = (
                matlab.double(df[num_cols].to_numpy(dtype=float).tolist()),
                df[datetime_key]
                .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                .where(df[datetime_key].notna(), ""),
                [
                    [
                        s if pd.notna(s) else ""
                        for s in df[str_cols].iloc[i].astype(str).tolist()
                    ]
                    for i in range(len(df))
                ],
            )
            num_mat_bgc, dt_str_bgc = (
                matlab.double(background[num_cols].to_numpy(dtype=float).tolist()),
                background[datetime_key]
                .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                .where(background[datetime_key].notna(), ""),
            )

            self.eng.workspace["strMat"] = str_mat
            self.eng.workspace["numMat"] = num_mat
            self.eng.workspace["numMatBackgound"] = num_mat_bgc
            self.eng.workspace["dtStrBackgound"] = dt_str_bgc.tolist()
            self.eng.workspace["dtStr"] = dt_str.tolist()
            self.eng.workspace["numCols"] = list(num_cols)
            self.eng.workspace["strCols"] = list(str_cols)
            self.eng.workspace["xlimStr"] = [
                x.strftime("%Y-%m-%d %H:%M:%S") for x in xlim
            ]
            self.eng.workspace["left_yparams"] = list(left_yparams)
            self.eng.workspace["right_yparams"] = list(right_yparams)
            self.eng.workspace["left_yparam_labels"] = list(left_param_labels)
            self.eng.workspace["right_yparam_labels"] = list(right_param_labels)
            self.eng.workspace["title_txt"] = title_txt
            self.eng.workspace["vlinesStr"] = [v.strftime("%Y-%m-%d %H:%M:%S") for v in vlines]
            self.eng.workspace["vline_styles"] = list(vline_styles)
            self.eng.workspace["draw_legend"] = data_dict.get("draw_legend", False)
            self.eng.workspace["xlabel_txt"] = data_dict.get("xlabel_txt", "Time, UT")
            self.eng.workspace["markers"] = data_dict.get("markers", ["-", "-"])
            self.eng.workspace["ms"] = data_dict.get("ms", 1)

            self.eng.eval(
                f"""
                    xlim = datetime(xlimStr,"InputFormat","yyyy-MM-dd HH:mm:ss");
                    vlines = datetime(vlinesStr,"InputFormat","yyyy-MM-dd HH:mm:ss");
                    T = array2table(numMat, "VariableNames", numCols);
                    T.datetime = datetime(dtStr(:), "InputFormat","yyyy-MM-dd HH:mm:ss.SSSSSS");
                    B = array2table(numMatBackgound, "VariableNames", numCols);
                    B.datetime = datetime(dtStrBackgound(:), "InputFormat","yyyy-MM-dd HH:mm:ss.SSSSSS");
                    [ax, tax] = sp.plot_TS(...
                            T, "datetime", left_yparams, right_yparams, ...
                            xlim=xlim, left_yparam_labels=left_yparam_labels, ...
                            right_yparam_labels=right_yparam_labels, ...
                            color_direction = "dark2light", ms=3, date_tick_format="HH", ...
                            title_txt=title_txt, txt_pos=[0.9 0.9], ...
                            vlines=vlines, vline_styles=vline_styles, ...
                            draw_legend=draw_legend, xlabel_txt=xlabel_txt, dual_frame=true ...
                    );

                    sp.plot_TS(...
                            B, "datetime", left_yparams, right_yparams, ...
                            xlim=xlim, left_yparam_labels=left_yparam_labels, ...
                            right_yparam_labels=right_yparam_labels, date_tick_format="HH", ...
                            color_direction = "dark2light", ms=ms, markers=markers, ...
                            title_txt=title_txt, txt_pos=[0.9 0.9], ...
                            vlines=vlines, vline_style=vline_styles, dual_frame=true,  ...
                            draw_legend=false, xlabel_txt=xlabel_txt, ax=ax ...
                    );
                """,
                nargout=0,
            )
        self.eng.eval(
            f"""
            sp.save(fullfile("{self.fig_path}", "{fig_file_name}"));
            sp.close();
            """,
            nargout=0,
        )
        return


    def generate_doppler_figure(
        self,
        data_dicts: list[dict],
        fig_file_name: str,
        fig_title: str = "",
        fontsize: int = 16,
        fig_shape: tuple = (8, 2),
    ):
        self.eng.eval(
            f"""
            sp = SaoSummaryPlots("{fig_title}", 3, {len(data_dicts)}, {fontsize}, [{fig_shape[0]} {fig_shape[1]}*{len(data_dicts)}]);
            """,
            nargout=0,
        )

        for i, data_dict in enumerate(data_dicts):
            (
                df,
                datetime_key,
                xlim,
                color_direction,
                title_txt,
                vlines,
                vline_styles,
            ) = (
                data_dict["dataset"],
                data_dict.get("datetime_key", "datetime"),
                data_dict.get("xlim", []),
                data_dict.get("color_direction", "dark2light"),
                data_dict.get("title_txt", f"({chr(65+i)})"),
                data_dict.get("vlines", []),
                data_dict.get("vline_styles", []),
            )

            num_cols, str_cols = (
                df.select_dtypes(include=["number"]).columns,
                df.columns.difference(df.select_dtypes(include=["number"]).columns),
            )

            # numeric as matlab.double
            num_mat, dt_str, str_mat = (
                matlab.double(df[num_cols].to_numpy(dtype=float).tolist()),
                df[datetime_key]
                .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                .where(df[datetime_key].notna(), ""),
                [
                    [
                        s if pd.notna(s) else ""
                        for s in df[str_cols].iloc[i].astype(str).tolist()
                    ]
                    for i in range(len(df))
                ],
            )

            self.eng.workspace["strMat"] = str_mat
            self.eng.workspace["numMat"] = num_mat
            self.eng.workspace["dtStr"] = dt_str.tolist()
            self.eng.workspace["numCols"] = list(num_cols)
            self.eng.workspace["strCols"] = list(str_cols)
            self.eng.workspace["xlimStr"] = [
                x.strftime("%Y-%m-%d %H:%M:%S") for x in xlim
            ]
            self.eng.workspace["title_txt"] = title_txt
            self.eng.workspace["vlinesStr"] = [v.strftime("%Y-%m-%d %H:%M:%S") for v in vlines]
            self.eng.workspace["vline_styles"] = list(vline_styles)
            self.eng.workspace["draw_legend"] = data_dict.get("draw_legend", False)
            self.eng.workspace["xlabel_txt"] = data_dict.get("xlabel_txt", "Time, UT")
            self.eng.workspace["ms"] = data_dict.get("ms", 6)

            self.eng.eval(
                f"""
                xlim = datetime(xlimStr,"InputFormat","yyyy-MM-dd HH:mm:ss");
                vlines = datetime(vlinesStr,"InputFormat","yyyy-MM-dd HH:mm:ss");
                T = array2table(numMat, "VariableNames", numCols);
                T.datetime = datetime(dtStr(:), "InputFormat","yyyy-MM-dd HH:mm:ss.SSSSSS");

                ax = sp.get_axes(false, {i+1});
                sp.plot_TS(...
                        T, "datetime", ["Vx"], [], ylabels=["$V_x$, m/s"], ...
                        xlim=xlim, left_yparam_labels=[], ...
                        right_yparam_labels=[], left_ylim=[-100 100],...
                        color_direction = "light2dark", ms=10, date_tick_format="HH", ...
                        title_txt=title_txt, txt_pos=[0.6 0.9], ax=ax,...
                        vlines=vlines, vline_styles=vline_styles, color_map=[255 0 0], ...
                        draw_legend=false, xlabel_txt="", left_axis_color="k" ...
                );
                ax = sp.get_axes(false, {i+3});
                sp.plot_TS(...
                        T, "datetime", ["Vy"], [], ylabels=["$V_y$, m/s"], ...
                        xlim=xlim, left_yparam_labels=[], ...
                        right_yparam_labels=[], left_ylim=[-100 100],...
                        color_direction = "light2dark", ms=10, date_tick_format="HH", ...
                        title_txt="", txt_pos=[0.6 0.9], color_map=[255 0 0], ...
                        vlines=vlines, vline_styles=vline_styles, ax=ax, ...
                        draw_legend=false, xlabel_txt="", left_axis_color="k" ...
                );
                ax = sp.get_axes(false, {i+5});
                sp.plot_TS(...
                        T, "datetime", ["Vz"], [], ylabels=["$V_z$, m/s"], ...
                        xlim=xlim, left_yparam_labels=[], ...
                        right_yparam_labels=[], left_ylim=[-20 20],...
                        color_direction = "light2dark", ms=10, date_tick_format="HH", ...
                        title_txt="", txt_pos=[0.6 0.9], color_map=[255 0 0], ...
                        vlines=vlines, vline_styles=vline_styles, ax=ax, ...
                        draw_legend=false, left_axis_color="k" ...
                );
                
                """,
                nargout=0,
            )
        self.eng.eval(
            f"""
            sp.save(fullfile("{self.fig_path}", "{fig_file_name}"));
            sp.close();
            """,
            nargout=0,
        )
        return


    def generate_skymap_figure(
        self,
        data_dicts: list[dict],
        fig_file_name: str,
        fig_title: str = "",
        fontsize: int = 16,
        fig_shape: tuple = (4, 4),
        nrows=4,
        ncols=4,
    ):
        self.eng.eval(
            f"""
            sp = SkySummaryPlots("{fig_title}", {nrows}, {ncols}, {fontsize}, [{fig_shape[0]*ncols} {fig_shape[1]*nrows}]);
            """,
            nargout=0,
        )
        for i, data_dict in enumerate(data_dicts):
            df = data_dict["dataset"]
            num_cols, str_cols = (
                df.select_dtypes(include=["number"]).columns,
                df.columns.difference(df.select_dtypes(include=["number"]).columns),
            )

            # numeric as matlab.double
            num_mat, str_mat = (
                matlab.double(df[num_cols].to_numpy(dtype=float).tolist()),
                [
                    [
                        s if pd.notna(s) else ""
                        for s in df[str_cols].iloc[i].astype(str).tolist()
                    ]
                    for i in range(len(df))
                ],
            )
            self.eng.workspace["strMat"] = str_mat
            self.eng.workspace["numMat"] = num_mat
            self.eng.workspace["numCols"] = list(num_cols)
            self.eng.workspace["strCols"] = list(str_cols)
            self.eng.workspace["cbar"] = data_dict.get("cbar", False)
            self.eng.workspace["tag_direction"] = data_dict.get("tag_direction", False)
            self.eng.workspace["text_txt"] = data_dict.get("text_txt", "")
            self.eng.eval(
                f"""
                T = array2table(numMat, "VariableNames", numCols);
                sp.plot_skymap(T, cbar=cbar, tag_direction=tag_direction, rlim=5, text_txt=text_txt);
                """,
                nargout=0,
            )
        self.eng.eval(
            f"""
            sp.save(fullfile("{self.fig_path}", "{fig_file_name}"));
            sp.close();
            """,
            nargout=0,
        )
        return