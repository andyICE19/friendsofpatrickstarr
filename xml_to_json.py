import json
import xmltodict

def load_xml(file_path, file_name):
    """
    Parameters: file path and file name
    Returns: ordered dictionary
    """
    with open(file_path+file_name) as xml_file:
        data_dict = xmltodict.parse(xml_file.read()) #Ordered dictionary

    data_element_list = data_dict['DataElementsList']['DataElement'] # List of dictionaries, each dictionary is a CDE

    return data_element_list

def keyparse(correct_keys, key_string): #def keyparse(correct_keys,key_string):
    """
    Parameters: 
    - correct keys: dictionary of <unparsed key>: <parsed key>
    - key_string: unparsed key in string
    Return: the right key values based on subset json file
    """
    return correct_keys[key_string]

def create_dictionary(full_dict, wanted_keys,correct_keys):
    """
    Function returns dictionary with only subset of data which are useful, while correcting the keys into desired format
    Parameters: full dictionary, keys that are needed (list) in the dictionary 
    Return: Dictionary with corrected key as key, and either list or another dictionary as value
    """
    return {keyparse(correct_keys,wanted_key): full_dict[wanted_key] for wanted_key in wanted_keys}


def clean_dictionary(data_element_list, list_of_keys_main, list_of_keys_concept, list_of_keys_value, list_of_keys_permissible_values,correct_keys):
    new_dict = {}

    for data_element in data_element_list:
        for (key, val_or_dict) in data_element.items():
            if key == 'PUBLICID':
                new_key = val_or_dict #make public id as key
                new_dict[new_key] = {} # initialise the dictionary within each cde public_id
            else: 
                if key in list_of_keys_main:
                    new_dict[new_key][keyparse(correct_keys,key)] = val_or_dict # to create dict of all elements
                elif key == 'DATAELEMENTCONCEPT':
                    concept_dict = create_dictionary(data_element[key], list_of_keys_concept,correct_keys) 
                    new_dict[new_key][keyparse(correct_keys,key)] = concept_dict
                elif key == 'VALUEDOMAIN':
                    value_dict = create_dictionary(data_element[key], list_of_keys_value,correct_keys)

                    if value_dict[keyparse(correct_keys,'PermissibleValues')] != None:
                        perm_value_list = value_dict[keyparse(correct_keys,'PermissibleValues')]['PermissibleValues_ITEM']

                        if type(perm_value_list) == list:
                            permissible_values = [create_dictionary(value,list_of_keys_permissible_values,correct_keys) for value in perm_value_list] # dictionary of valid_value and public id per valid value

                        else:
                            permissible_values = [create_dictionary(perm_value_list, list_of_keys_permissible_values,correct_keys)]       
                            
                        value_dict[keyparse(correct_keys,'PermissibleValues')] = permissible_values
                    
                    for value_dict_header in value_dict: #Replace all null values to None
                        if value_dict[value_dict_header] == {'@NULL':'TRUE'}:
                            value_dict[value_dict_header] = None

                    if value_dict[keyparse(correct_keys,'PermissibleValues')] == None: #dictionary of permissble_values should be an empty list, not a NoneType
                        value_dict[keyparse(correct_keys,'PermissibleValues')] = []

                    new_dict[new_key][keyparse(correct_keys,key)] = value_dict
    return new_dict

