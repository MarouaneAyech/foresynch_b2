import torch.nn as nn


def diagnose_trainable(model: nn.Module, head: nn.Module | None = None,
                        lora_class=None, verbose: bool = True) -> dict:
    """Diagnostic des paramètres entraînables par couche. Fonctionne pour tout
    scénario : full FT, FT sélectif, LoRA, full LoRA, hybride."""
    if verbose:
        print("── Diagnostic des paramètres entraînables ──\n")
        print(f"{'Module':<14}{'Total':>10}{'Train':>10}{'LoRA':>10}{'Statut':>16}")
        print("-" * 60)

    tot_all = tot_train = tot_lora = 0

    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        if n_params == 0:
            continue
        n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)

        n_lora = 0
        if lora_class is not None:
            for sub in module.modules():
                if isinstance(sub, lora_class):
                    n_lora += sum(
                        p.numel() for n, p in sub.named_parameters()
                        if p.requires_grad and "lora" in n
                    )

        if verbose:
            if n_train == 0:
                status = "gele"
            elif n_train == n_params:
                status = "FT complet"
            elif n_lora > 0 and n_lora == n_train:
                status = "gele+LoRA"
            else:
                status = "mixte"
            print(f"  {name:<12s}{n_params/1e6:>9.3f}M{n_train/1e6:>9.3f}M"
                  f"{n_lora/1e6:>9.3f}M{status:>16}")

        tot_all += n_params
        tot_train += n_train
        tot_lora += n_lora

    if head is not None:
        h_params = sum(p.numel() for p in head.parameters())
        h_train = sum(p.numel() for p in head.parameters() if p.requires_grad)
        if verbose:
            h_status = "entrainable" if h_train == h_params else ("gele" if h_train == 0 else "mixte")
            print(f'  {"head":<12s}{h_params/1e6:>9.3f}M{h_train/1e6:>9.3f}M{"":>9}{h_status:>16}')
        tot_all += h_params
        tot_train += h_train

    if verbose:
        print("-" * 60)
        print(f'  {"TOTAL":<12s}{tot_all/1e6:>9.3f}M{tot_train/1e6:>9.3f}M{tot_lora/1e6:>9.3f}M')
        print(f"\n  Entrainables : {tot_train/1e6:.3f}M / {tot_all/1e6:.3f}M "
              f"({100*tot_train/tot_all:.2f}%)")

    return {"total": tot_all, "trainable": tot_train, "lora": tot_lora}
