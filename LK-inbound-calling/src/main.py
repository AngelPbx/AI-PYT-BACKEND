from livekit_api import create_inbound_trunk, list_inbound_trunks, create_dispatch_rule

if __name__ == "__main__":
    print("Creating inbound SIP trunk...")
    trunk = create_inbound_trunk("Test Trunk", ["+18333659442"])
    print("Trunk:", trunk)

    trunk_id = trunk.get("trunk", {}).get("trunk_id")
    if trunk_id:
        print("\nCreating dispatch rule for room 'test-room'...")
        rule = create_dispatch_rule("test-room", [trunk_id])
        print("Dispatch Rule:", rule)

    print("\nListing all SIP trunks...")
    all_trunks = list_inbound_trunks()
    print(all_trunks)
