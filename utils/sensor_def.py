locations = {
    "lab": [
        {"serial": "f1230765", "crop": ((750, 1600), (200, 1000))},
        {"serial": "f1231723", "crop": ((750, 1750), (400, 1050))},
    ],
    "wall": [
        {"serial": "f1231723", "crop": None},  # stage rear right
        {"serial": "f1230814", "crop": None},  # stage front right        
        {"serial": "f0271635", "crop": None},  # stage front left
        {"serial": "f0221612", "crop": None},  #  stage rear left
    ]
#        {"serial": "f0245195", "crop": None},  # this one seems to be borked...
}

boards = {
    "canvas": {"x_verts": 4, "y_verts": 12, "square_size": 0.15},
    "poster": {"x_verts": 5, "y_verts": 9, "square_size": 0.08},
    "paper": {"x_verts": 3, "y_verts": 4, "square_size": 0.0585}
}

