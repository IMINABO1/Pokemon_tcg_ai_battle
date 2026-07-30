"""Script to explore card database from cg.api."""
import sys
from pathlib import Path

# Ensure cg is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "sample_submission" / "sample_submission"))

from cg.api import all_card_data, all_attack, CardType, EnergyType

def main():
    cards = all_card_data()
    attacks = all_attack()
    attack_by_id = {a.attackId: a for a in attacks}

    print(f"Total cards: {len(cards)}")
    print(f"Total attacks: {len(attacks)}")

    # Find ACE SPEC cards
    ace_specs = [c for c in cards if c.aceSpec]
    print("\n--- ACE SPEC Cards ---")
    for c in ace_specs:
        print(f"ID: {c.cardId}, Name: {c.name}, Type: {CardType(c.cardType).name}")

    # Find Basic Energy IDs
    print("\n--- Basic Energies ---")
    basic_energies = [c for c in cards if c.cardType == CardType.BASIC_ENERGY]
    for c in basic_energies:
        print(f"ID: {c.cardId}, Name: {c.name}, EnergyType: {EnergyType(c.energyType).name}")

    # Search for specific Pokemon (e.g., Ceruledge, Charcadet, Palafin, Finizen)
    target_names = ["Ceruledge", "Charcadet", "Palafin", "Finizen", "Judge", "Boss's Orders", "Lillie's Determination", "Cyrano", "Maximum Belt"]
    print("\n--- Targeted Card Search ---")
    for target in target_names:
        matches = [c for c in cards if target.lower() in c.name.lower()]
        print(f"\nQuery: '{target}' -> {len(matches)} matches:")
        for c in matches:
            atk_str = ""
            if c.attacks:
                atks = [attack_by_id.get(a_id) for a_id in c.attacks if a_id in attack_by_id]
                atk_str = ", ".join([f"{a.name} (dmg={a.damage}, cost={[EnergyType(e).name for e in a.energies]})" for a in atks if a])
            skills_str = ", ".join([f"{s.name}: {s.text}" for s in c.skills]) if c.skills else ""
            print(f"  ID={c.cardId} | Name='{c.name}' | Type={CardType(c.cardType).name} | HP={c.hp} | Basic={c.basic} Stage1={c.stage1} Stage2={c.stage2} ex={c.ex} | EvolvesFrom={c.evolvesFrom} | Attacks=[{atk_str}] | Skills=[{skills_str}]")


if __name__ == "__main__":
    main()
