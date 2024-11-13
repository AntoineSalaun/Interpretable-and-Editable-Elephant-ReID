class SEEK:
    def __init__(self, code):
        """
        Initialize the class with a list of SEEK codes.
        """
        #'B00T__E8000-0000X0_S00'

        if len(code) != 22:
            raise ValueError("wrongly-sized SEEK code ! It is + ", len(code), " characters long, but should be 23 characters long")
        if code[3] != 'T':
            raise ValueError("wrongly formatted SEEK code, missing a T in position 4")
        if code[6] != 'E':    
            raise ValueError("wrongly formatted SEEK code, missing an E in position 7")
        if code[11] != '-':    
            raise ValueError("wrongly formatted SEEK code, missing a - in position 12")
        if code[16] != 'X':
            raise ValueError("wrongly formatted SEEK code, missing an X in position 17")
        if code[19] != 'S':
            raise ValueError("wrongly formatted SEEK code, missing an S in position 20")

        # General
        self.sex = code[0]
        self.age = code[1] + code[2] #To-do : make sure that they always join as strings and not add as number (if inputed as number)
        # Tusks
        self.R_tusk = code[4]
        self.L_tusk = code[5]
        # Ears
        self.R_tear_1 = code[7]
        self.R_hole_1 = code[8]
        self.R_tear_2 = code[9]
        self.R_hole_2 = code[10]
        #-
        self.L_tear_1 = code[12]
        self.L_hole_1 = code[13]
        self.L_tear_2 = code[14]
        self.L_hole_2 = code[15]
        # Extreme
        self.R_extreme = code[17]
        self.L_extreme = code[18]
        # Special
        self.ear_special = code[20]
        self.body_special = code[21]

        self.SEEK_list_format = [self.sex, self.age, self.R_tusk, self.L_tusk, self.R_tear_1, self.R_hole_1, self.R_tear_2, self.R_hole_2, self.L_tear_1, self.L_hole_1, self.L_tear_2, self.L_hole_2, self.R_extreme, self.L_extreme, self.ear_special, self.body_special]


    def __getitem__(self, index):
        """
        Access SEEK codes by index. Such that each index corresponds to a meaningful part of the SEEK code.
        """
        return self.SEEK_list_format[index]

    def __str__(self):
        """
        Print representation of the SEEK.
        """
        return f"{self.sex}{self.age}T{self.R_tusk}{self.L_tusk}E{self.R_tear_1}{self.R_hole_1}{self.R_tear_2}{self.R_hole_2}-{self.L_tear_1}{self.L_hole_1}{self.L_tear_2}{self.L_hole_2}X{self.R_extreme}{self.L_extreme}S{self.ear_special}{self.body_special}"

