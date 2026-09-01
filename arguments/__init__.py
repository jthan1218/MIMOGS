from argparse import ArgumentParser, Namespace
import os
import sys

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            t = type(value)
            value = value if not fill_none else None

            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, "-" + key[0], default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, "-" + key[0], default=value, type=t
                    )

            else:
                if t == bool:
                    group.add_argument(
                        "--" + key, default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, default=value, type=t
                    )

    def extract(self, args):
        group = GroupParams()
        for k, v in vars(args).items():
            if k in vars(self) or ("_" + k) in vars(self):
                setattr(group, k, v)
        return group


class ModelParams(ParamGroup):
    """
    MIMOGS model/data level arguments
    """

    def __init__(self, parser: ArgumentParser, sentinel: bool = False):
        self._source_path = "./dataset/indoor_63by63"
        self._model_path = ""
        self.data_device = "cuda"
        self.eval = False

        # Array shape overrides, (horizontal, vertical) per side.  0 means the
        # shape is derived from the beam count found in the dataset, which is
        # the square factorization.  Set these only for a non-square UPA.
        self.rx_shape_h = 21
        self.rx_shape_v = 3
        self.tx_shape_h = 21
        self.tx_shape_v = 3

        # Beam-grid mode.  "dft" (default) keeps the DFT beam centers derived
        # from the (horizontal, vertical) array shape above and reproduces
        # every DeepMIMO result exactly.  "custom_angles" places the beam
        # centers at a measured analog steering codebook instead; the beam
        # count is then len(az) * len(el) and the array shape is unused.
        self.beam_grid_mode = "custom_angles"

        # Comma-separated steering angles in degrees for "custom_angles".
        # Empty means the measured 60 GHz defaults (21 azimuth x 3 elevation).
        self.beam_az_deg = ""
        self.beam_el_deg = ""

        self.init_mode = "random"
        self.vertices_path = ""
        self.max_active_rx_beams = 14
        self.max_active_tx_beams = 14
        self.renormalize_local_beam_weights = True

        # Renderer/training settings. Integers are used instead of bools
        # so both 0 and 1 can be supplied through the existing ParamGroup.
        self.batch_size = 8
        self.num_workers = 0
        self.num_epochs = 1000
        self.target_gaussians = 25_000
        self.use_cuda_rasterizer = 1
        self.use_amp = 0

        # Tie the Tx-side 3D covariance to the Rx-side one. With 1 the two
        # anchors share a single (scaling, rotation) pair, which reproduces the
        # previous shared-covariance behaviour exactly. With 0 each anchor
        # carries its own covariance and the two ends of a primitive are tied
        # only through the shared per-primitive gain.
        self.tie_covariance = 0

        # Number of Fourier frequency bands the DynamicGainNet's positional
        # encoding uses for each of its three 3-D inputs (xyz, rx, xyz-rx).
        # 6 is the historical value and keeps every existing run bit-identical;
        # 0 feeds the raw coordinates through instead (include_input stays
        # True), so the gain MLP sees 3 x 3 = 9 features instead of 3 x 39.
        self.gain_pe_frequencies = 6

        super().__init__(parser, "Model Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        if getattr(g, "source_path", ""):
            g.source_path = os.path.abspath(g.source_path)
        if getattr(g, "model_path", ""):
            g.model_path = os.path.abspath(g.model_path)
        if getattr(g, "vertices_path", ""):
            g.vertices_path = os.path.abspath(g.vertices_path)
        return g

class OptimizationParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        self.iterations = 0
        self.position_lr_init = 0.003
        self.position_lr_final = 0.000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 0

        self.opacity_lr = 0.025
        # self.scaling_lr = 0.005
        self.scaling_lr = 0.003
        # self.rotation_lr = 0.001
        self.rotation_lr = 0.0005
        self.optimizer_type = "default"

        # self.gain_lr = 0.0025
        self.opacity_lr_final = 0.003
        # self.gain_lr_final = 0.0003

        # 추가
        self.dynamic_gain_lr = 0.001
        self.dynamic_gain_lr_final = 0.0001

        # Anchor-tie regularizer weight: pulls the per-Gaussian Tx anchor
        # (_xyz_tx) toward the Rx anchor (_xyz). With a large default the
        # model behaves as a single-anchor (LoS / single-bounce) renderer;
        # the optimizer is free to separate the two anchors where the data
        # demands it (multi-bounce paths).
        self.lambda_anchor = 0

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser: ArgumentParser):

    args_cmdline = parser.parse_args(sys.argv[1:])
    cfgfile_string = "Namespace()"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath, "r", encoding="utf-8") as cfg_file:
            print("Config file found:", cfgfilepath)
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError, AttributeError):
        print("Config file not found.")
        pass

    args_cfgfile = eval(cfgfile_string, {"Namespace": Namespace}, {})
    merged_dict = vars(args_cfgfile).copy()

    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    return Namespace(**merged_dict)