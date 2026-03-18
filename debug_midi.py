from music21 import converter

midi = converter.parse("data/mz_330_1.mid")
print(f"Elements in flat: {len(midi.flat)}")
for i, el in enumerate(midi.flat[:20]):
    print(f"{i}: {el}")

print("\nParts:")
for i, part in enumerate(midi.parts):
    print(f"Part {i}: {part.id}, elements: {len(part.flat)}")
    for j, el in enumerate(part.flat[:10]):
         print(f"  {j}: {el}")
