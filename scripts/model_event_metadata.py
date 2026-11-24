"""Model-facing dates and weekly ATP singles ranks for four mixed-date main draws.

Start dates are local calendar dates on the men's singles draws or ATP media notes:
Washington: https://www.atptour.com/-/media/c858b0cae9c041e9bc0a951eeae48bd1.pdf
Winston-Salem: https://www.protennislive.com/posting/2024/6242/mds.pdf
Belgrade: https://www.protennislive.com/posting/2024/4787/mds.pdf
Metz: https://www.protennislive.com/posting/2024/341/mds.pdf

Weekly ranking excerpts: Jeff Sackmann (CC BY-NC-SA 4.0), archived by Aneeshers at
https://github.com/Aneeshers/tennis-sackmann-archive/tree/83733587353df8a41f2fd4f516147d5aa83f5a8d/atp
https://creativecommons.org/licenses/by-nc-sa/4.0/
All four draws use the last published Monday ranking strictly before their opening day.
Look up the archive player ID in atp_rankings_10s.csv or atp_rankings_20s.csv
on the specified ranking_date. The comments beside ranks give those IDs.
All draw entrants matched uniquely to atp_players.csv by name and nationality.
For Y09V the archive names Bu Yunchaokete and the draw names Yunchaokete Bu.
See https://www.atptour.com/en/players/bu-yunchaokete/y09v/rankings-history.

SHA-256 for the fixed source CSVs:
atp_players.csv      89e81bb3a02561dee29afe53dbcdabe4ae22f725fb5d0303617366a57e3b7a83
atp_rankings_10s.csv 2921c7ccc1f2c3709bab2a2570ad18de4ca49b126e6b078a53e0dc2eaa6fa8de
atp_rankings_20s.csv d527934459a6322fbbce24f99434f93b3f5a986ee65c7a542f6096d478c7c42a
"""


# Event ID -> (official main-draw start, pre-event ranking date).
EVENT_METADATA = {
    "2019-418": ("20190729", "20190722"),
    "2024-6242": ("20240818", "20240812"),
    "2024-4787": ("20241103", "20241028"),
    "2024-341": ("20241103", "20241028"),
}

# Weekly ATP ranks keyed by the provider's official alphanumeric ATP player IDs.
RANKS_BY_WEEK = {
    "20190722": {
        "AG37": 23,  # Felix Auger-Aliassime; archive ID 200000
        "BE90": 75,  # Ricardas Berankis; archive ID 105575
        "BK92": 71,  # Alexander Bublik; archive ID 122330
        "C977": 17,  # Marin Cilic; archive ID 105227
        "CA99": 84,  # Marius Copil; archive ID 105657
        "DH58": 34,  # Alex de Minaur; archive ID 200282
        "E687": 55,  # Daniel Evans; archive ID 105554
        "E690": 100,  # Matthew Ebden; archive ID 105051
        "E831": 35,  # Kyle Edmund; archive ID 106378
        "E865": 184,  # Christopher Eubanks; archive ID 111153
        "F811": 123,  # Bjorn Fratangelo; archive ID 106216
        "G967": 122,  # Peter Gojowczyk; archive ID 105376
        "GB88": 18,  # David Goffin; archive ID 105676
        "GB93": 132,  # Norbert Gombos; archive ID 105613
        "H996": 39,  # Pierre-Hugues Herbert; archive ID 105732
        "HB71": 44,  # Hubert Hurkacz; archive ID 128034
        "HG86": 82,  # Lloyd Harris; archive ID 144750
        "I186": 14,  # John Isner; archive ID 104545
        "I305": 126,  # Ilya Ivashka; archive ID 125802
        "J267": 103,  # Malek Jaziri; archive ID 104291
        "K336": 81,  # Ivo Karlovic; archive ID 103333
        "KA98": 94,  # Bradley Klahn; archive ID 105614
        "KB09": 101,  # Denis Kudla; archive ID 106045
        "KD35": 208,  # Thai-Son Kwiatkowski; archive ID 111515
        "KE17": 48,  # Nick Kyrgios; archive ID 106401
        "KE29": 8,  # Karen Khachanov; archive ID 111575
        "KI95": 66,  # Miomir Kecmanovic; archive ID 200175
        "ME82": 64,  # Adrian Mannarino; archive ID 105173
        "MM58": 9,  # Daniil Medvedev; archive ID 106421
        "N732": 73,  # Yoshihito Nishioka; archive ID 106415
        "O522": 57,  # Reilly Opelka; archive ID 124187
        "P09Z": 95,  # Alexei Popyrin; archive ID 200615
        "PD31": 28,  # Benoit Paire; archive ID 105332
        "PG94": 179,  # Marc Polmans; archive ID 117360
        "PL56": 128,  # Tommy Paul; archive ID 126205
        "R975": 21,  # Milos Raonic; archive ID 105683
        "RE44": 78,  # Andrey Rublev; archive ID 126094
        "SD32": 33,  # Gilles Simon; archive ID 104468
        "SF62": 282,  # Tim Smyczek; archive ID 105065
        "SL28": 36,  # Jan-Lennard Struff; archive ID 105526
        "SM25": 180,  # Jack Sock; archive ID 106058
        "SP24": 97,  # Brayden Schnur; archive ID 111790
        "T786": 68,  # Jo-Wilfried Tsonga; archive ID 104542
        "TC12": 177,  # Mikael Torpegaard; archive ID 111453
        "TC61": 43,  # Jordan Thompson; archive ID 111442
        "TD51": 41,  # Frances Tiafoe; archive ID 126207
        "TE51": 6,  # Stefanos Tsitsipas; archive ID 126774
        "Y124": 190,  # Donald Young; archive ID 105385
    },
    "20240812": {
        "B0BI": 20,  # Sebastian Baez; archive ID 202104
        "BT72": 39,  # Nuno Borges; archive ID 132686
        "BU13": 85,  # Zizou Bergs; archive ID 200267
        "C0AU": 28,  # Francisco Cerundolo; archive ID 202103
        "C0DF": 108,  # Francisco Comesana; archive ID 207681
        "C0H0": 93,  # Arthur Cazaux; archive ID 209070
        "CD85": 513,  # Pablo Carreno Busta; archive ID 105807
        "CE77": 80,  # Federico Coria; archive ID 105948
        "CG80": 83,  # Borna Coric; archive ID 106432
        "D0FJ": 34,  # Luciano Darderi; archive ID 209260
        "D994": 67,  # James Duckworth; archive ID 105902
        "DA81": 90,  # Taro Daniel; archive ID 106121
        "DB63": 106,  # Laslo Djere; archive ID 111513
        "E687": 178,  # Daniel Evans; archive ID 105554
        "E865": 123,  # Christopher Eubanks; archive ID 111153
        "EA24": 37,  # Tomas Martin Etcheverry; archive ID 144869
        "F724": 84,  # Marton Fucsovics; archive ID 105916
        "G09O": 61,  # Hugo Gaston; archive ID 200384
        "GB88": 92,  # David Goffin; archive ID 105676
        "H0BH": 65,  # Rinky Hijikata; archive ID 208014
        "K09F": 63,  # Pavel Kotov; archive ID 200303
        "K0EE": 429,  # Omni Kumar; archive ID 208300
        "KG27": 495,  # Strong Kirchheimer; archive ID 126177
        "LB66": 88,  # Constant Lestienne; archive ID 106005
        "M0AW": 236,  # Nicolas Mejia; archive ID 200711
        "M0CI": 50,  # Fabian Marozsan; archive ID 206681
        "M0QI": 57,  # Alex Michelsen; archive ID 210506
        "ME82": 33,  # Adrian Mannarino; archive ID 105173
        "MP20": 77,  # Alexandre Muller; archive ID 124186
        "MU94": 89,  # Jaume Munar; archive ID 144719
        "MW02": 71,  # Corentin Moutet; archive ID 144895
        "N0BS": 38,  # Mariano Navone; archive ID 208363
        "N897": 74,  # Sumit Nagal; archive ID 111576
        "O483": 86,  # Christopher O'Connell; archive ID 106331
        "O522": 322,  # Reilly Opelka; archive ID 124187
        "RC91": 59,  # Arthur Rinderknech; archive ID 126239
        "S0H2": 60,  # Alexander Shevchenko; archive ID 207686
        "S0K7": 110,  # Zachary Svajda; archive ID 208260
        "S0LA": 190,  # Dominic Stricker; archive ID 208502
        "S0N0": 198,  # Tristan Schoolkate; archive ID 209262
        "S0RE": 78,  # Juncheng Shang; archive ID 209992
        "SU87": 56,  # Lorenzo Sonego; archive ID 132283
        "SX50": 66,  # Roman Safiullin; archive ID 126128
        "SX91": 70,  # Thiago Seyboth Wild; archive ID 205734
        "T0HA": 232,  # Learner Tien; archive ID 210530
        "V812": 75,  # Botic van de Zandschulp; archive ID 122298
        "V832": 97,  # Aleksandar Vukic; archive ID 126846
        "W09E": 95,  # Adam Walton; archive ID 200443
    },
    "20241028": {
        "A0CJ": 117,  # Duje Ajdukovic; archive ID 207213
        "AE14": 82,  # Daniel Altmaier; archive ID 127157
        "BD06": 46,  # Roberto Bautista Agut; archive ID 105138
        "BK24": 155,  # Gregoire Barrere; archive ID 106296
        "BM95": 126,  # Benjamin Bonzi; archive ID 126127
        "BT72": 34,  # Nuno Borges; archive ID 132686
        "BU13": 65,  # Zizou Bergs; archive ID 200267
        "C0AU": 29,  # Francisco Cerundolo; archive ID 202103
        "C977": 201,  # Marin Cilic; archive ID 105227
        "CF59": 54,  # Roberto Carballes Baena; archive ID 106148
        "D0C1": 109,  # Jesper de Jong; archive ID 207411
        "D0DW": 173,  # Titouan Droguet; archive ID 208169
        "D0FJ": 43,  # Luciano Darderi; archive ID 209260
        "D0HJ": 779,  # Branko Djuric; archive ID 210025
        "D994": 77,  # James Duckworth; archive ID 105902
        "DB63": 129,  # Laslo Djere; archive ID 111513
        "EA24": 40,  # Tomas Martin Etcheverry; archive ID 144869
        "F510": 79,  # Fabio Fognini; archive ID 104926
        "F724": 88,  # Marton Fucsovics; archive ID 105916
        "G09O": 81,  # Hugo Gaston; archive ID 200384
        "G628": 133,  # Richard Gasquet; archive ID 104755
        "GH33": 260,  # Manuel Guinard; archive ID 131905
        "H996": 124,  # Pierre-Hugues Herbert; archive ID 105732
        "HB64": 90,  # Quentin Halys; archive ID 111460
        "K09F": 73,  # Pavel Kotov; archive ID 200303
        "K0AZ": 99,  # Aleksandar Kovacevic; archive ID 206499
        "KI63": 146,  # Lukas Klein; archive ID 127760
        "KI95": 55,  # Miomir Kecmanovic; archive ID 200175
        "L0BV": 30,  # Jiri Lehecka; archive ID 208103
        "L987": 72,  # Dusan Lajovic; archive ID 105583
        "M0CI": 57,  # Fabian Marozsan; archive ID 206681
        "M0G4": 107,  # Harold Mayot; archive ID 208004
        "M0JF": 158,  # Hamad Medjedovic; archive ID 209098
        "M0QI": 44,  # Alex Michelsen; archive ID 210506
        "ME82": 58,  # Adrian Mannarino; archive ID 105173
        "MJ08": 100,  # Thiago Monteiro; archive ID 106329
        "MO44": 41,  # Pedro Martinez; archive ID 124079
        "MP20": 68,  # Alexandre Muller; archive ID 124186
        "MW02": 70,  # Corentin Moutet; archive ID 144895
        "N0AE": 35,  # Brandon Nakashima; archive ID 206909
        "N0BS": 45,  # Mariano Navone; archive ID 208363
        "N771": 56,  # Cameron Norrie; archive ID 111815
        "N897": 84,  # Sumit Nagal; archive ID 111576
        "O483": 71,  # Christopher O'Connell; archive ID 106331
        "P0K5": 715,  # Theo Papamalamis; archive ID 210494
        "RE44": 7,  # Andrey Rublev; archive ID 126094
        "RH16": 8,  # Casper Ruud; archive ID 134770
        "S0H2": 63,  # Alexander Shevchenko; archive ID 207686
        "S0RE": 51,  # Juncheng Shang; archive ID 209992
        "SL28": 42,  # Jan-Lennard Struff; archive ID 105526
        "SU55": 80,  # Denis Shapovalov; archive ID 133430
        "SU87": 52,  # Lorenzo Sonego; archive ID 132283
        "SX50": 59,  # Roman Safiullin; archive ID 126128
        "V0DZ": 114,  # Luca Van Assche; archive ID 209414
        "W367": 149,  # Stan Wawrinka; archive ID 104527
        "Y09V": 74,  # Yunchaokete Bu; archive ID 207352
    },
}


def corrected_tour_row(row):
    """Return corrected date/ranks for audited main draws, leaving the source row intact."""
    event = row["tourney_id"]
    if event not in EVENT_METADATA or row["round"] in {"Q1", "Q2", "Q3", "Q4"}:
        return row
    start, ranking_week = EVENT_METADATA[event]
    ranks = RANKS_BY_WEEK[ranking_week]
    effective = dict(row, tourney_date=start)
    for side in ("winner", "loser"):
        player_id = row[f"{side}_id"]
        if player_id not in ranks:
            raise ValueError(f"Missing verified {ranking_week} rank for {player_id} in {event}.")
        effective[f"{side}_rank"] = str(ranks[player_id])
    return effective
