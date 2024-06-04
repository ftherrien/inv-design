def test_generate():
    from didgen import generate
    import shutil

    config = {'n_data': 100, 'num_epochs': 100}
    
    out = generate(2, "test", config, seed=1)

    assert out[0]["smiles"] == '[H]OC1=C2N=C(N([H])[H])N=C(N([H])[H])N2N2N([H])N2O1'
    assert out[1]["smiles"] == '[H]C1(N=O)OC2(C1([H])[H])C([H])([H])C2([H])N=O'

    assert out[0]["property"][0] == 4.329557418823242
    assert out[1]["property"][0] == 4.355047225952148

    shutil.rmtree("test")
