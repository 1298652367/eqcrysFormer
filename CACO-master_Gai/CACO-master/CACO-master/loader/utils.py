import torch


def get_db_info():
    """Get DB info."""
    db_info = {
        # https://doi.org/10.6084/m9.figshare.6815705
        "dft_2d": [
            "https://ndownloader.figshare.com/files/26808917",
            "d2-3-12-2021.json",
            "Obtaining 2D dataset 1.1k ...",
            "https://www.nature.com/articles/s41524-020-00440-1",
        ],
        # https://doi.org/10.6084/m9.figshare.6815699
        "dft_3d": [
            "https://ndownloader.figshare.com/files/29204826",
            "jdft_3d-8-18-2021.json",
            "Obtaining 3D dataset 55k ...",
            "https://www.nature.com/articles/s41524-020-00440-1",
        ],
        # https://doi.org/10.6084/m9.figshare.6815699
        "cfid_3d": [
            "https://ndownloader.figshare.com/files/29205201",
            "cfid_3d-8-18-2021.json",
            "Obtaining 3D dataset 55k ...",
            "https://www.nature.com/articles/s41524-020-00440-1",
        ],
        # https://doi.org/10.6084/m9.figshare.14213522
        "jff": [
            "https://ndownloader.figshare.com/files/28937793",
            # "https://ndownloader.figshare.com/files/26809760",
            "jff-7-24-2021.json",
            # "jff-3-12-2021.json",
            "Obtaining JARVIS-FF 2k ...",
            "https://www.nature.com/articles/s41524-020-00440-1",
        ],
        "mp_3d_2020": [
            "https://ndownloader.figshare.com/files/26791259",
            "all_mp.json",
            "Obtaining Materials Project-3D CFID dataset 127k...",
            "https://doi.org/10.1063/1.4812323",
        ],
        # https://doi.org/10.6084/m9.figshare.14177630
        "megnet": [
            "https://ndownloader.figshare.com/files/26724977",
            "megnet.json",
            "Obtaining MEGNET-3D CFID dataset 69k...",
            "https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294",
        ],
        # https://doi.org/10.6084/m9.figshare.14745435
        "megnet2": [
            "https://ndownloader.figshare.com/files/28332741",
            "megnet-mp-2019-04-01.json",
            "Obtaining MEGNET-3D CFID dataset 133k...",
            "https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294",
        ],
        # https://doi.org/10.6084/m9.figshare.14745327
        "edos_pdos": [
            "https://ndownloader.figshare.com/files/29216859",
            "edos-up_pdos-elast_interp-8-18-2021.json",
            "Interpolated electronic total dos spin-up dataset 55k...",
            "https://www.nature.com/articles/s41524-020-00440-1",
        ],
        # https://doi.org/10.6084/m9.figshare.13054247
        "mp_3d": [
            "https://ndownloader.figshare.com/files/24979850",
            "CFID_mp_desc_data_84k.json",
            "Obtaining Materials Project-3D CFID dataset 84k...",
            "https://doi.org/10.1063/1.4812323",
        ],
        # https://doi.org/10.6084/m9.figshare.13055333
        "oqmd_3d": [
            "https://ndownloader.figshare.com/files/24981170",
            "CFID_OQMD_460k.json",
            "Obtaining OQMD-3D CFID dataset 460k...",
            "https://www.nature.com/articles/npjcompumats201510",
        ],
        # https://doi.org/10.6084/m9.figshare.14206169
        "oqmd_3d_no_cfid": [
            "https://ndownloader.figshare.com/files/26790182",
            "all_oqmd.json",
            "Obtaining OQMD-3D  dataset 800k...",
            "https://www.nature.com/articles/npjcompumats201510",
        ],
        # https://doi.org/10.6084/m9.figshare.14205083
        "twod_matpd": [
            "https://ndownloader.figshare.com/files/26789006",
            "twodmatpd.json",
            "Obtaining 2DMatPedia dataset 6k...",
            "https://www.nature.com/articles/s41597-019-0097-3",
        ],
        # https://doi.org/10.6084/m9.figshare.14213603
        "polymer_genome": [
            "https://ndownloader.figshare.com/files/26809907",
            "pgnome.json",
            "Obtaining Polymer genome 1k...",
            "https://www.nature.com/articles/sdata201612",
        ],
        "qm9_std_jctc": [
            "https://ndownloader.figshare.com/files/28715319",
            "qm9_std_jctc.json",
            "Obtaining QM9 standardized dataset 130k,"
            + "From https://doi.org/10.1021/acs.jctc.7b00577,+",
            "https://www.nature.com/articles/sdata201422",
        ],
        # https://doi.org/10.6084/m9.figshare.14827584
        # Use qm9_std_jctc instaed
        "qm9_dgl": [
            "https://ndownloader.figshare.com/files/28541196",
            "qm9_dgl.json",
            "Obtaining QM9 dataset 130k, from DGL...",
            "https://www.nature.com/articles/sdata201422",
        ],
        # https://doi.org/10.6084/m9.figshare.14912820.v1
        "cod": [
            "https://ndownloader.figshare.com/files/28715301",
            "cod_db.json",
            "Obtaining COD dataset 431k",
            "https://doi.org/10.1107/S1600576720016532",
        ],
        # Use qm9_std_jctc instaed
        "qm9": [
            "https://ndownloader.figshare.com/files/27627596",
            "qm9_data_cfid.json",
            "Obtaining QM9 dataset 134k...",
            "https://www.nature.com/articles/sdata201422",
        ],
        # https://doi.org/10.6084/m9.figshare.15127788
        "qe_tb": [
            "https://ndownloader.figshare.com/files/29070555",
            "jqe_tb_folder.json",
            "Obtaining QETB dataset 860k...",
            "https://arxiv.org/abs/2112.11585",
        ],
        # https://doi.org/10.6084/m9.figshare.14812050
        "omdb": [
            "https://ndownloader.figshare.com/files/28501761",
            "omdbv1.json",
            "Obtaining OMDB dataset 12.5k...",
            "https://doi.org/10.1002/qute.201900023",
        ],
        # https://doi.org/10.6084/m9.figshare.14812044
        "qmof": [
            "https://figshare.com/ndownloader/files/30972640",
            "qmof_db.json",
            "Obtaining QMOF dataset 20k...",
            "https://www.cell.com/matter/fulltext/S2590-2385(21)00070-9",
        ],
        # https://doi.org/10.6084/m9.figshare.15127758
        "hmof": [
            "https://figshare.com/ndownloader/files/30972655",
            "hmof_db_9_18_2021.json",
            "Obtaining hMOF dataset 137k...",
            "https://doi.org/10.1021/acs.jpcc.6b08729",
        ],
        # https://figshare.com/account/projects/100325/articles/14960157
        "c2db": [
            "https://ndownloader.figshare.com/files/28682010",
            "c2db_atoms.json",
            "Obtaining C2DB dataset 3.5k...",
            "https://iopscience.iop.org/article/10.1088/2053-1583/aacfc1",
        ],
        # https://figshare.com/account/projects/100325/articles/14962356
        "hopv": [
            "https://ndownloader.figshare.com/files/28814184",
            "hopv_15.json",
            "Obtaining HOPV15 dataset 4.5k...",
            "https://www.nature.com/articles/sdata201686",
        ],
        # https://figshare.com/account/projects/100325/articles/14962356
        "pdbbind_core": [
            "https://ndownloader.figshare.com/files/28874802",
            "pdbbind_2015_core.json",
            "Obtaining PDBBind dataset 195...",
            "https://doi.org/10.1093/bioinformatics/btu626",
        ],
        # https://doi.org/10.6084/m9.figshare.14812038
        "pdbbind": [
            "https://ndownloader.figshare.com/files/28816368",
            "pdbbind_2015.json",
            "Obtaining PDBBind dataset 11k...",
            "https://doi.org/10.1093/bioinformatics/btu626",
        ],
        # https://doi.org/10.6084/m9.figshare.13215308
        "aflow2": [
            "https://ndownloader.figshare.com/files/25453265",
            "CFID_AFLOW2.json",
            "Obtaining AFLOW-2 CFID dataset 400k...",
            "https://doi.org/10.1016/j.commatsci.2012.02.005",
        ],
        # https://doi.org/10.6084/m9.figshare.14211860
        "arXiv": [
            "https://ndownloader.figshare.com/files/26804795",
            "arXivdataset.json",
            "Obtaining arXiv dataset 1.8 million...",
            "https://www.kaggle.com/Cornell-University/arxiv",
        ],
        # https://doi.org/10.6084/m9.figshare.14211857
        "cord19": [
            "https://ndownloader.figshare.com/files/26804798",
            "cord19.json",
            "Obtaining CORD19 dataset 223k...",
            "https://github.com/usnistgov/cord19-cdcs-nist",
        ],
        # https://doi.org/10.6084/m9.figshare.13154159
        "raw_files": [
            "https://ndownloader.figshare.com/files/25295732",
            "figshare_data-10-28-2020.json",
            "Obtaining raw io files 145k...",
            "https://www.nature.com/articles/s41524-020-00440-1",
        ],
    }
    return db_info


def compute_angle(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["offset"]
    r2 = edges.dst["offset"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1) + 1e-6
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"angle": bond_cosine.float()}


