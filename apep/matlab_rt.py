import matlab.engine as engine
import os

def setup_matlab_engine(
    pharlap_dir:str = "/home/chakras4/Research/CodeBase/raytrace/pharlap/pharlap_4.5.3",
    close: bool = True,
    R12 = 100, kp=0, doppler_flag=0,
    irregs_flag=0, rb=0,
):
    os.system(f"export DIR_MODELS_REF_DAT={pharlap_dir}/dat")

    eng = engine.start_matlab()
    eng.cd("/".join(pharlap_dir.split("/")[:-1]))
    eng.addpath(pharlap_dir + "/src/matlab/")
    print(eng.pwd())
    eng.close("all")

    # Initialize
    eng.workspace["R12"] = R12
    eng.workspace["doppler_flag"] = doppler_flag
    eng.workspace["irregs_flag"] = irregs_flag
    eng.workspace["kp"] = kp

    
    if close:
        eng.quit()
        return 0
    else:
        return eng

if __name__ =="__main__":
    setup_matlab_engine()