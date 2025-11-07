# -*- coding: utf-8 -*-
from elftools.elf.elffile import ELFFile
from elftools.common.exceptions import ELFParseError
from elftools.common.exceptions import ELFError
import pandas as pd
import os


class ELFMiner(object):
    def __init__(self, binary_path):
        self.binary_path = binary_path
        
        self.section_names = ['.text', '.bss', '.comment', '.data', '.data1', '.debug', '.dynamic', '.dynstr', \
                              '.dynsym', '.fini', '.hash', '.init', '.got', '.interp', '.line', '.note', '.plt', '.rodata', \
                              '.rodata1', '.shstrtab', '.strtab', '.symtab', '.sdata', '.sbss', '.lit8', '.gptab', '.conflict', \
                               '.tdesc', '.lit4', '.reginfo', '.liblist', '.rel.dyn', '.rel.plt', '.got.plt']
            
        self.header_feature = 0
        self.section_header_feature = 0
        self.program_header_feature = 0
        self.symbol_section_feature = 0
        self.dynamic_section_feature = 0
        self.dynamic_symbol_section_feature = 0
        self.relocation_section_feature = 0
        self.global_offeset_table_feature = 0
        self.hash_table_feature = 0
        
        self.feature_name_order = ['endianness', 'elf_class', 'version', 'elf_type', 'machine', 'entry_point', 'flags', 'elf_header_size', 'program_header_size', 
                   'program_header_num', 'section_header_size', 'section_header_num', 'section_name_index', '.text_section_type', '.text_section_flags', 
                   '.text_section_size', '.text_section_link', '.text_section_info', '.text_section_addr_align', '.bss_section_type', '.bss_section_flags', 
                   '.bss_section_size', '.bss_section_link', '.bss_section_info', '.bss_section_addr_align', '.comment_section_type', '.comment_section_flags', 
                   '.comment_section_size', '.comment_section_link', '.comment_section_info', '.comment_section_addr_align', '.data_section_type', 
                   '.data_section_flags', '.data_section_size', '.data_section_link', '.data_section_info', '.data_section_addr_align', '.data1_section_type', 
                   '.data1_section_flags', '.data1_section_size', '.data1_section_link', '.data1_section_info', '.data1_section_addr_align', '.debug_section_type', 
                   '.debug_section_flags', '.debug_section_size', '.debug_section_link', '.debug_section_info', '.debug_section_addr_align', '.dynamic_section_type', 
                   '.dynamic_section_flags', '.dynamic_section_size', '.dynamic_section_link', '.dynamic_section_info', '.dynamic_section_addr_align', 
                   '.dynstr_section_type', '.dynstr_section_flags', '.dynstr_section_size', '.dynstr_section_link', '.dynstr_section_info', '.dynstr_section_addr_align', 
                   '.dynsym_section_type', '.dynsym_section_flags', '.dynsym_section_size', '.dynsym_section_link', '.dynsym_section_info', '.dynsym_section_addr_align', 
                   '.fini_section_type', '.fini_section_flags', '.fini_section_size', '.fini_section_link', '.fini_section_info', '.fini_section_addr_align', 
                   '.hash_section_type', '.hash_section_flags', '.hash_section_size', '.hash_section_link', '.hash_section_info', '.hash_section_addr_align', 
                   '.init_section_type', '.init_section_flags', '.init_section_size', '.init_section_link', '.init_section_info', '.init_section_addr_align', 
                   '.got_section_type', '.got_section_flags', '.got_section_size', '.got_section_link', '.got_section_info', '.got_section_addr_align', 
                   '.interp_section_type', '.interp_section_flags', '.interp_section_size', '.interp_section_link', '.interp_section_info', '.interp_section_addr_align', 
                   '.line_section_type', '.line_section_flags', '.line_section_size', '.line_section_link', '.line_section_info', '.line_section_addr_align', 
                   '.note_section_type', '.note_section_flags', '.note_section_size', '.note_section_link', '.note_section_info', '.note_section_addr_align', 
                   '.plt_section_type', '.plt_section_flags', '.plt_section_size', '.plt_section_link', '.plt_section_info', '.plt_section_addr_align', 
                   '.rodata_section_type', '.rodata_section_flags', '.rodata_section_size', '.rodata_section_link', '.rodata_section_info', '.rodata_section_addr_align', 
                   '.rodata1_section_type', '.rodata1_section_flags', '.rodata1_section_size', '.rodata1_section_link', '.rodata1_section_info', 
                   '.rodata1_section_addr_align', '.shstrtab_section_type', '.shstrtab_section_flags', '.shstrtab_section_size', '.shstrtab_section_link', 
                   '.shstrtab_section_info', '.shstrtab_section_addr_align', '.strtab_section_type', '.strtab_section_flags', '.strtab_section_size', 
                   '.strtab_section_link', '.strtab_section_info', '.strtab_section_addr_align', '.symtab_section_type', '.symtab_section_flags', '.symtab_section_size', 
                   '.symtab_section_link', '.symtab_section_info', '.symtab_section_addr_align', '.sdata_section_type', '.sdata_section_flags', '.sdata_section_size', 
                   '.sdata_section_link', '.sdata_section_info', '.sdata_section_addr_align', '.sbss_section_type', '.sbss_section_flags', '.sbss_section_size', 
                   '.sbss_section_link', '.sbss_section_info', '.sbss_section_addr_align', '.lit8_section_type', '.lit8_section_flags', '.lit8_section_size', 
                   '.lit8_section_link', '.lit8_section_info', '.lit8_section_addr_align', '.gptab_section_type', '.gptab_section_flags', '.gptab_section_size', 
                   '.gptab_section_link', '.gptab_section_info', '.gptab_section_addr_align', '.conflict_section_type', '.conflict_section_flags', 
                   '.conflict_section_size', '.conflict_section_link', '.conflict_section_info', '.conflict_section_addr_align', '.tdesc_section_type', 
                   '.tdesc_section_flags', '.tdesc_section_size', '.tdesc_section_link', '.tdesc_section_info', '.tdesc_section_addr_align', '.lit4_section_type', 
                   '.lit4_section_flags', '.lit4_section_size', '.lit4_section_link', '.lit4_section_info', '.lit4_section_addr_align', '.reginfo_section_type', 
                   '.reginfo_section_flags', '.reginfo_section_size', '.reginfo_section_link', '.reginfo_section_info', '.reginfo_section_addr_align', 
                   '.liblist_section_type', '.liblist_section_flags', '.liblist_section_size', '.liblist_section_link', '.liblist_section_info', 
                   '.liblist_section_addr_align', '.rel.dyn_section_type', '.rel.dyn_section_flags', '.rel.dyn_section_size', '.rel.dyn_section_link', 
                   '.rel.dyn_section_info', '.rel.dyn_section_addr_align', '.rel.plt_section_type', '.rel.plt_section_flags', '.rel.plt_section_size', 
                   '.rel.plt_section_link', '.rel.plt_section_info', '.rel.plt_section_addr_align', '.got.plt_section_type', '.got.plt_section_flags', 
                   '.got.plt_section_size', '.got.plt_section_link', '.got.plt_section_info', '.got.plt_section_addr_align', '0_segment_type', '0_file_size', 
                   '0_memory_size', '0_flags', '0_alignment', '1_segment_type', '1_file_size', '1_memory_size', '1_flags', '1_alignment', '2_segment_type', 
                   '2_file_size', '2_memory_size', '2_flags', '2_alignment', '3_segment_type', '3_file_size', '3_memory_size', '3_flags', '3_alignment', 
                   '4_segment_type', '4_file_size', '4_memory_size', '4_flags', '4_alignment', '5_segment_type', '5_file_size', '5_memory_size', '5_flags', '5_alignment', 
                   '6_segment_type', '6_file_size', '6_memory_size', '6_flags', '6_alignment', '7_segment_type', '7_file_size', '7_memory_size', '7_flags', '7_alignment', 
                   '8_segment_type', '8_file_size', '8_memory_size', '8_flags', '8_alignment', 'symtab_total_symbols', 'symtab_local_symbols', 'symtab_global_symbols', 
                   'symtab_weak_symbols', 'symtab_stb_lo_proc_symbols', 'symtab_stb_hi_proc_symbols', 'symtab_stt_notype', 'symtab_local_objects', 'symtab_global_objects', 
                   'symtab_weak_objects', 'symtab_local_functions', 'symtab_global_functions', 'symtab_weak_functions', 'symtab_sections', 'symtab_files', 
                   'symtab_stt_lo_proc_objects', 'symtab_stt_hi_proc_objects', 'total_entries', 'null_entries', 'needed_libraries', 'plt_relocation_size', 'hiproc_entries',
                   'plt_got_entries', 'hash_table_entries', 'string_table_entries', 'symbol_table_entries', 'rela_entries', 'rela_size', 'rela_entry_size', 
                   'string_table_size', 'symbol_table_entry_size', 'init_function_address', 'fini_function_address', 'soname_entries', 'rpath_entries', 
                   'symbolic_entries', 'rel_entries', 'rel_size', 'rel_entry_size', 'plt_relocation_type', 'debug_entries', 'text_relocation_entries', 'jmprel_entries', 
                   'loproc_entries', 'dynsym_total_symbols', 'dynsym_local_symbols', 'dynsym_global_symbols', 'dynsym_weak_symbols', 'dynsym_stb_lo_proc_symbols', 
                   'dynsym_stb_hi_proc_symbols', 'dynsym_stt_notype', 'dynsym_local_objects', 'dynsym_global_objects', 'dynsym_weak_objects', 'dynsym_local_functions', 
                   'dynsym_global_functions', 'dynsym_weak_functions', 'dynsym_sections', 'dynsym_files', 'dynsym_stt_lo_proc_objects', 'dynsym_stt_hi_proc_objects', 
                   'got_size', 'hash_table_size']

    def file_check(self):
        with open(self.binary_path, 'rb') as f:
            try:
                elffile = ELFFile(f)
                return True
            except Exception as e:
                return False
            
    def elf_header(self):
        with open(self.binary_path, 'rb') as f:
            elffile = ELFFile(f)
            header = elffile.header
            elf_header_info = {
                'endianness': None,       # ELFENDIAN
                'elf_class': None,        # ELFHEICLASS
                'version': None,          # ELFHEIVERSION
                'elf_type': None,         # ELFHEETYPE
                'machine': None,          # ELFHEMACHINE
                'entry_point': None,      # ELFHEENTRY
                'flags': None,            # ELFHEFLAGS
                'elf_header_size': None,  # ELFHEESIZE
                'program_header_size': None,  # ELFHEPHENTSIZE
                'program_header_num': None,   # ELFHEPHNUM
                'section_header_size': None,  # ELFHEPHENTSIZE
                'section_header_num': None,   # ELFHEPHNUM
                'section_name_index': None    # ELFHEPHSTRINDX
            }
        
            # Extract the required fields from the ELF header
            elf_header_info['endianness'] = header['e_ident']['EI_DATA']  # ELFENDIAN
            elf_header_info['elf_class'] = header['e_ident']['EI_CLASS']  # ELFHEICLASS
            elf_header_info['version'] = header['e_version']  # ELFHEIVERSION
            elf_header_info['elf_type'] = header['e_type']  # ELFHEETYPE
            elf_header_info['machine'] = header['e_machine']  # ELFHEMACHINE
            elf_header_info['entry_point'] = header['e_entry']  # ELFHEENTRY
            elf_header_info['flags'] = header['e_flags']  # ELFHEFLAGS
            elf_header_info['elf_header_size'] = header['e_ehsize']  # ELFHEESIZE
            elf_header_info['program_header_size'] = header['e_phentsize']  # ELFHEPHENTSIZE
            elf_header_info['program_header_num'] = header['e_phnum']  # ELFHEPHNUM
            elf_header_info['section_header_size'] = header['e_shentsize']  # ELFHEPHENTSIZE
            elf_header_info['section_header_num'] = header['e_shnum']  # ELFHEPHNUM
            elf_header_info['section_name_index'] = header['e_shstrndx']  # ELFHEPHSTRINDX
            return elf_header_info
    
    def section_header(self):      
        with open(self.binary_path, 'rb') as f:
            elffile = ELFFile(f)   
            section_header_feature = {}
            for section_name in self.section_names:
                section_info = {                  
                    section_name+'_section_type': None,
                    section_name+'_section_flags': None,
                    section_name+'_section_size': None,
                    section_name+'_section_link': None,
                    section_name+'_section_info': None,
                    section_name+'_section_addr_align': None
                }
                section_header_feature.update(section_info)
            try:    
                exist_section_names = [section.name for section in elffile.iter_sections()]
                for section_name in self.section_names:
                    if section_name in exist_section_names:
                        section = elffile.get_section_by_name(section_name)
                        section_header_feature[section_name+'_section_type'] = section['sh_type']
                        section_header_feature[section_name+'_section_flags'] = section['sh_flags']
                        section_header_feature[section_name+'_section_size'] = section['sh_size']
                        section_header_feature[section_name+'_section_link'] = section['sh_link']
                        section_header_feature[section_name+'_section_info'] = section['sh_info']
                        section_header_feature[section_name+'_section_addr_align'] = section['sh_addralign']                      
            except Exception as e:
                print(f"An error occurred: {e}")
            return section_header_feature
    
    def program_header(self):       
        with open(self.binary_path, 'rb') as f:
            elffile = ELFFile(f)         
            program_header_feature = {}
            for i, segment in enumerate(elffile.iter_segments()):
                if i > 8: 
                    break               
                segment_info = {
                    str(i)+'_segment_type': segment['p_type'],
                    str(i)+'_file_size': segment['p_filesz'],
                    str(i)+'_memory_size': segment['p_memsz'],
                    str(i)+'_flags': segment['p_flags'],
                    str(i)+'_alignment': segment['p_align']
                }               
                program_header_feature.update(segment_info)                        
            if i<8:
                for j in range(i+1, 9):
                    segment_info = {
                        str(j)+'_segment_type': segment['p_type'],
                        str(j)+'_file_size': segment['p_filesz'],
                        str(j)+'_memory_size': segment['p_memsz'],
                        str(j)+'_flags': segment['p_flags'],
                        str(j)+'_alignment': segment['p_align']
                    }               
                    program_header_feature.update(segment_info)       
            # print(f"{program_header_feature.keys()}, {i}")
            return program_header_feature
        
    def symbol_table(self):
        with open(self.binary_path, 'rb') as f:
            elffile = ELFFile(f)
            symbol_section_feature = {
                'symtab_total_symbols': None,
                'symtab_local_symbols': None,
                'symtab_global_symbols': None,
                'symtab_weak_symbols': None,
                'symtab_stb_lo_proc_symbols': None,
                'symtab_stb_hi_proc_symbols': None,
                'symtab_stt_notype': None,
                'symtab_local_objects': None,
                'symtab_global_objects': None,
                'symtab_weak_objects': None,
                'symtab_local_functions': None,
                'symtab_global_functions': None,
                'symtab_weak_functions': None,
                'symtab_sections': None,
                'symtab_files': None,
                'symtab_stt_lo_proc_objects': None,
                'symtab_stt_hi_proc_objects': None}   
            try:
                symtab = elffile.get_section_by_name('.symtab')           
            
                if symtab is not None:                                   
                    for symbol in symtab.iter_symbols():
                        st_info = symbol['st_info']
                        bind = st_info.bind
                        type = st_info.type
                        
                        # Update the counts, initializing with 0 if None
                        if symbol_section_feature['symtab_total_symbols'] is None:
                            symbol_section_feature['symtab_total_symbols'] = 0
                        symbol_section_feature['symtab_total_symbols'] += 1
                        
                        if bind == 'STB_LOCAL':
                            if symbol_section_feature['symtab_local_symbols'] is None:
                                symbol_section_feature['symtab_local_symbols'] = 0
                            symbol_section_feature['symtab_local_symbols'] += 1
                        elif bind == 'STB_GLOBAL':
                            if symbol_section_feature['symtab_global_symbols'] is None:
                                symbol_section_feature['symtab_global_symbols'] = 0
                            symbol_section_feature['symtab_global_symbols'] += 1
                        elif bind == 'STB_WEAK':
                            if symbol_section_feature['symtab_weak_symbols'] is None:
                                symbol_section_feature['symtab_weak_symbols'] = 0
                            symbol_section_feature['symtab_weak_symbols'] += 1
                        elif bind == 'STB_LOPROC':
                            if symbol_section_feature['symtab_stb_lo_proc_symbols'] is None:
                                symbol_section_feature['symtab_stb_lo_proc_symbols'] = 0
                            symbol_section_feature['symtab_stb_lo_proc_symbols'] += 1
                        elif bind == 'STB_HIPROC':
                            if symbol_section_feature['symtab_stb_hi_proc_symbols'] is None:
                                symbol_section_feature['symtab_stb_hi_proc_symbols'] = 0
                            symbol_section_feature['symtab_stb_hi_proc_symbols'] += 1
                        
                        if type == 'STT_OBJECT':
                            if bind == 'STB_LOCAL':
                                if symbol_section_feature['symtab_local_objects'] is None:
                                    symbol_section_feature['symtab_local_objects'] = 0
                                symbol_section_feature['symtab_local_objects'] += 1
                            elif bind == 'STB_GLOBAL':
                                if symbol_section_feature['symtab_global_objects'] is None:
                                    symbol_section_feature['symtab_global_objects'] = 0
                                symbol_section_feature['symtab_global_objects'] += 1
                            elif bind == 'STB_WEAK':
                                if symbol_section_feature['symtab_weak_objects'] is None:
                                    symbol_section_feature['symtab_weak_objects'] = 0
                                symbol_section_feature['symtab_weak_objects'] += 1
                        elif type == 'STT_FUNC':
                            if bind == 'STB_LOCAL':
                                if symbol_section_feature['symtab_local_functions'] is None:
                                    symbol_section_feature['symtab_local_functions'] = 0
                                symbol_section_feature['symtab_local_functions'] += 1
                            elif bind == 'STB_GLOBAL':
                                if symbol_section_feature['symtab_global_functions'] is None:
                                    symbol_section_feature['symtab_global_functions'] = 0
                                symbol_section_feature['symtab_global_functions'] += 1
                            elif bind == 'STB_WEAK':
                                if symbol_section_feature['symtab_weak_functions'] is None:
                                    symbol_section_feature['symtab_weak_functions'] = 0
                                symbol_section_feature['symtab_weak_functions'] += 1
                        elif type == 'STT_SECTION':
                            if symbol_section_feature['symtab_sections'] is None:
                                symbol_section_feature['symtab_sections'] = 0
                            symbol_section_feature['symtab_sections'] += 1
                        elif type == 'STT_FILE':
                            if symbol_section_feature['symtab_files'] is None:
                                symbol_section_feature['symtab_files'] = 0
                            symbol_section_feature['symtab_files'] += 1
                        elif type == 'STT_LOPROC':
                            if symbol_section_feature['symtab_stt_lo_proc_objects'] is None:
                                symbol_section_feature['symtab_stt_lo_proc_objects'] = 0
                            symbol_section_feature['symtab_stt_lo_proc_objects'] += 1
                        elif type == 'STT_HIPROC':
                            if symbol_section_feature['symtab_stt_hi_proc_objects'] is None:
                                symbol_section_feature['symtab_stt_hi_proc_objects'] = 0
                            symbol_section_feature['symtab_stt_hi_proc_objects'] += 1     
                        else:
                            if symbol_section_feature['symtab_stt_notype'] is None:
                                symbol_section_feature['symtab_stt_notype'] = 0
                            symbol_section_feature['symtab_stt_notype'] += 1     
            except Exception as e:
                print(f"An error occurred: {e}")
            return symbol_section_feature

    def dynamic_symbol(self):
        with open(self.binary_path, 'rb') as f:
            elffile = ELFFile(f)
            dynamic_categories = {
            'total_entries': None,
            'null_entries': None,
            'needed_libraries': None,
            'plt_relocation_size': None,
            'hiproc_entries': None,
            'plt_got_entries': None,
            'hash_table_entries': None,
            'string_table_entries': None,
            'symbol_table_entries': None,
            'rela_entries': None,
            'rela_size': None,
            'rela_entry_size': None,
            'string_table_size': None,
            'symbol_table_entry_size': None,
            'init_function_address': None,
            'fini_function_address': None,
            'soname_entries': None,
            'rpath_entries': None,
            'symbolic_entries': None,
            'rel_entries': None,
            'rel_size': None,
            'rel_entry_size': None,
            'plt_relocation_type': None,
            'debug_entries': None,
            'text_relocation_entries': None,
            'jmprel_entries': None,
            'loproc_entries': None
            }
            
            try: 
                dynamic = elffile.get_section_by_name('.dynamic')
                
                if dynamic is not None: 
                    for entry in dynamic.iter_tags():
                        tag = entry.entry.d_tag
                        
                        if tag == 'DT_NULL':
                            if dynamic_categories['null_entries'] is None:
                                dynamic_categories['null_entries'] = 0
                            dynamic_categories['null_entries'] += 1
                        elif tag == 'DT_NEEDED':
                            if dynamic_categories['needed_libraries'] is None:
                                dynamic_categories['needed_libraries'] = 0
                            dynamic_categories['needed_libraries'] += 1
                        elif tag == 'DT_PLTRELSZ':
                            if dynamic_categories['plt_relocation_size'] is None:
                                dynamic_categories['plt_relocation_size'] = 0
                            dynamic_categories['plt_relocation_size'] += 1
                        elif tag == 'DT_HIPROC':
                            if dynamic_categories['hiproc_entries'] is None:
                                dynamic_categories['hiproc_entries'] = 0
                            dynamic_categories['hiproc_entries'] += 1
                        elif tag == 'DT_PLTGOT':
                            if dynamic_categories['plt_got_entries'] is None:
                                dynamic_categories['plt_got_entries'] = 0
                            dynamic_categories['plt_got_entries'] += 1
                        elif tag == 'DT_HASH':
                            if dynamic_categories['hash_table_entries'] is None:
                                dynamic_categories['hash_table_entries'] = 0
                            dynamic_categories['hash_table_entries'] += 1
                        elif tag == 'DT_STRTAB':
                            if dynamic_categories['string_table_entries'] is None:
                                dynamic_categories['string_table_entries'] = 0
                            dynamic_categories['string_table_entries'] += 1
                        elif tag == 'DT_SYMTAB':
                            if dynamic_categories['symbol_table_entries'] is None:
                                dynamic_categories['symbol_table_entries'] = 0
                            dynamic_categories['symbol_table_entries'] += 1
                        elif tag == 'DT_RELA':
                            if dynamic_categories['rela_entries'] is None:
                                dynamic_categories['rela_entries'] = 0
                            dynamic_categories['rela_entries'] += 1
                        elif tag == 'DT_RELASZ':
                            if dynamic_categories['rela_size'] is None:
                                dynamic_categories['rela_size'] = 0
                            dynamic_categories['rela_size'] += 1
                        elif tag == 'DT_RELAENT':
                            if dynamic_categories['rela_entry_size'] is None:
                                dynamic_categories['rela_entry_size'] = 0
                            dynamic_categories['rela_entry_size'] += 1
                        elif tag == 'DT_STRSZ':
                            if dynamic_categories['string_table_size'] is None:
                                dynamic_categories['string_table_size'] = 0
                            dynamic_categories['string_table_size'] += 1
                        elif tag == 'DT_SYMENT':
                            if dynamic_categories['symbol_table_entry_size'] is None:
                                dynamic_categories['symbol_table_entry_size'] = 0
                            dynamic_categories['symbol_table_entry_size'] += 1
                        elif tag == 'DT_INIT':
                            if dynamic_categories['init_function_address'] is None:
                                dynamic_categories['init_function_address'] = 0
                            dynamic_categories['init_function_address'] += 1
                        elif tag == 'DT_FINI':
                            if dynamic_categories['fini_function_address'] is None:
                                dynamic_categories['fini_function_address'] = 0
                            dynamic_categories['fini_function_address'] += 1
                        elif tag == 'DT_SONAME':
                            if dynamic_categories['soname_entries'] is None:
                                dynamic_categories['soname_entries'] = 0
                            dynamic_categories['soname_entries'] += 1
                        elif tag == 'DT_RPATH':
                            if dynamic_categories['rpath_entries'] is None:
                                dynamic_categories['rpath_entries'] = 0
                            dynamic_categories['rpath_entries'] += 1
                        elif tag == 'DT_SYMBOLIC':
                            if dynamic_categories['symbolic_entries'] is None:
                                dynamic_categories['symbolic_entries'] = 0
                            dynamic_categories['symbolic_entries'] += 1
                        elif tag == 'DT_REL':
                            if dynamic_categories['rel_entries'] is None:
                                dynamic_categories['rel_entries'] = 0
                            dynamic_categories['rel_entries'] += 1
                        elif tag == 'DT_RELSZ':
                            if dynamic_categories['rel_size'] is None:
                                dynamic_categories['rel_size'] = 0
                            dynamic_categories['rel_size'] += 1
                        elif tag == 'DT_RELENT':
                            if dynamic_categories['rel_entry_size'] is None:
                                dynamic_categories['rel_entry_size'] = 0
                            dynamic_categories['rel_entry_size'] += 1
                        elif tag == 'DT_PLTREL':
                            if dynamic_categories['plt_relocation_type'] is None:
                                dynamic_categories['plt_relocation_type'] = 0
                            dynamic_categories['plt_relocation_type'] += 1
                        elif tag == 'DT_DEBUG':
                            if dynamic_categories['debug_entries'] is None:
                                dynamic_categories['debug_entries'] = 0
                            dynamic_categories['debug_entries'] += 1
                        elif tag == 'DT_TEXTREL':
                            if dynamic_categories['text_relocation_entries'] is None:
                                dynamic_categories['text_relocation_entries'] = 0
                            dynamic_categories['text_relocation_entries'] += 1
                        elif tag == 'DT_JMPREL':
                            if dynamic_categories['jmprel_entries'] is None:
                                dynamic_categories['jmprel_entries'] = 0
                            dynamic_categories['jmprel_entries'] += 1
                        elif tag == 'DT_LOPROC':
                            if dynamic_categories['loproc_entries'] is None:
                                dynamic_categories['loproc_entries'] = 0
                            dynamic_categories['loproc_entries'] += 1    
                    # Count the total number of dynamic entries
                    dynamic_categories['total_entries'] = sum(value for key, value in dynamic_categories.items() if value is not None)
            except Exception as e:
                print(f"An error occurred: {e}")
            dynamic_section_feature = dynamic_categories
            return dynamic_section_feature
        
    def dynamic_symbol_section(self):
        with open(self.binary_path, 'rb') as f:
           elffile = ELFFile(f)  
           dynamic_symbol_counts = {
               'dynsym_total_symbols': None,
               'dynsym_local_symbols': None,
               'dynsym_global_symbols': None,
               'dynsym_weak_symbols': None,
               'dynsym_stb_lo_proc_symbols': None,
               'dynsym_stb_hi_proc_symbols': None,
               'dynsym_stt_notype': None,
               'dynsym_local_objects': None,
               'dynsym_global_objects': None,
               'dynsym_weak_objects': None,
               'dynsym_local_functions': None,
               'dynsym_global_functions': None,
               'dynsym_weak_functions': None,
               'dynsym_sections': None,
               'dynsym_files': None,
               'dynsym_stt_lo_proc_objects': None,
               'dynsym_stt_hi_proc_objects': None
           }
           
           try: 
               dynsym = elffile.get_section_by_name('.dynsym')                     
               if dynsym is not None:           
                   for symbol in dynsym.iter_symbols():
                       st_info = symbol['st_info']
                       bind = st_info.bind
                       type = st_info.type
                       
                       # Update the counts, initializing with 0 if None
                       if dynamic_symbol_counts['dynsym_total_symbols'] is None:
                           dynamic_symbol_counts['dynsym_total_symbols'] = 0
                       dynamic_symbol_counts['dynsym_total_symbols'] += 1
                       
                       if bind == 'STB_LOCAL':
                           if dynamic_symbol_counts['dynsym_local_symbols'] is None:
                               dynamic_symbol_counts['dynsym_local_symbols'] = 0
                           dynamic_symbol_counts['dynsym_local_symbols'] += 1
                       elif bind == 'STB_GLOBAL':
                           if dynamic_symbol_counts['dynsym_global_symbols'] is None:
                               dynamic_symbol_counts['dynsym_global_symbols'] = 0
                           dynamic_symbol_counts['dynsym_global_symbols'] += 1
                       elif bind == 'STB_WEAK':
                           if dynamic_symbol_counts['dynsym_weak_symbols'] is None:
                               dynamic_symbol_counts['dynsym_weak_symbols'] = 0
                           dynamic_symbol_counts['dynsym_weak_symbols'] += 1
                       elif bind == 'STB_LOPROC':
                           if dynamic_symbol_counts['dynsym_stb_lo_proc_symbols'] is None:
                               dynamic_symbol_counts['dynsym_stb_lo_proc_symbols'] = 0
                           dynamic_symbol_counts['dynsym_stb_lo_proc_symbols'] += 1
                       elif bind == 'STB_HIPROC':
                           if dynamic_symbol_counts['dynsym_stb_hi_proc_symbols'] is None:
                               dynamic_symbol_counts['dynsym_stb_hi_proc_symbols'] = 0
                           dynamic_symbol_counts['dynsym_stb_hi_proc_symbols'] += 1
                       
                       if type == 'STT_OBJECT':
                           if bind == 'STB_LOCAL':
                               if dynamic_symbol_counts['dynsym_local_objects'] is None:
                                   dynamic_symbol_counts['dynsym_local_objects'] = 0
                               dynamic_symbol_counts['dynsym_local_objects'] += 1
                           elif bind == 'STB_GLOBAL':
                               if dynamic_symbol_counts['dynsym_global_objects'] is None:
                                   dynamic_symbol_counts['dynsym_global_objects'] = 0
                               dynamic_symbol_counts['dynsym_global_objects'] += 1
                           elif bind == 'STB_WEAK':
                               if dynamic_symbol_counts['dynsym_weak_objects'] is None:
                                   dynamic_symbol_counts['dynsym_weak_objects'] = 0
                               dynamic_symbol_counts['dynsym_weak_objects'] += 1
                       elif type == 'STT_FUNC':
                           if bind == 'STB_LOCAL':
                               if dynamic_symbol_counts['dynsym_local_functions'] is None:
                                   dynamic_symbol_counts['dynsym_local_functions'] = 0
                               dynamic_symbol_counts['dynsym_local_functions'] += 1
                           elif bind == 'STB_GLOBAL':
                               if dynamic_symbol_counts['dynsym_global_functions'] is None:
                                   dynamic_symbol_counts['dynsym_global_functions'] = 0
                               dynamic_symbol_counts['dynsym_global_functions'] += 1
                           elif bind == 'STB_WEAK':
                               if dynamic_symbol_counts['dynsym_weak_functions'] is None:
                                   dynamic_symbol_counts['dynsym_weak_functions'] = 0
                               dynamic_symbol_counts['dynsym_weak_functions'] += 1
                       elif type == 'STT_SECTION':
                           if dynamic_symbol_counts['dynsym_sections'] is None:
                               dynamic_symbol_counts['dynsym_sections'] = 0
                           dynamic_symbol_counts['dynsym_sections'] += 1
                       elif type == 'STT_FILE':
                           if dynamic_symbol_counts['dynsym_files'] is None:
                               dynamic_symbol_counts['dynsym_files'] = 0
                           dynamic_symbol_counts['dynsym_files'] += 1
                       elif type == 'STT_LOPROC':
                           if dynamic_symbol_counts['dynsym_stt_lo_proc_objects'] is None:
                               dynamic_symbol_counts['dynsym_stt_lo_proc_objects'] = 0
                           dynamic_symbol_counts['dynsym_stt_lo_proc_objects'] += 1
                       elif type == 'STT_HIPROC':
                           if dynamic_symbol_counts['dynsym_stt_hi_proc_objects'] is None:
                               dynamic_symbol_counts['dynsym_stt_hi_proc_objects'] = 0
                           dynamic_symbol_counts['dynsym_stt_hi_proc_objects'] += 1
                       else:
                           if dynamic_symbol_counts['dynsym_stt_notype'] is None:
                               dynamic_symbol_counts['dynsym_stt_notype'] = 0
                           dynamic_symbol_counts['dynsym_stt_notype'] += 1    
           except Exception as e:
               print(f"An error occurred: {e}")
           dynamic_symbol_section_feature = dynamic_symbol_counts
           return dynamic_symbol_section_feature
       
    def global_offset_table(self):      
        with open(self.binary_path, 'rb') as f:
            elffile = ELFFile(f)
            
            # Initialize the feature dictionary with None for GOT size
            got_features = {
                'got_size': None
            }
            
            try:
                # Iterate over all sections to find the GOT section
                for section in elffile.iter_sections():
                    if section.name == '.got' or section.name == '.got.plt':
                        # The size of the GOT section is the size in bytes
                        got_features['got_size'] = section['sh_size']
                        break  # Stop after finding the first GOT section
            except Exception as e:
                print(f"An error occurred: {e}")
            global_offeset_table_feature = got_features
            return global_offeset_table_feature
        
    def hash_table(self):
        with open(self.binary_path, 'rb') as f:
            elffile = ELFFile(f)
            
            # Initialize the feature dictionary with None for hash table size
            hash_features = {
                'hash_table_size': None
            }
            
            try:
                # Iterate over all sections to find the hash table section
                for section in elffile.iter_sections():
                    if section.name == '.hash' or section.name == '.gnu.hash':
                        # The size of the hash table is the size in bytes, so we convert it to the number of entries
                        # assuming each entry is 4 bytes (32 bits)
                        hash_features['hash_table_size'] = section['sh_size'] // 4
                        break  # Stop after finding the first hash table section
            except Exception as e:
                print(f"An error occurred: {e}")
            hash_table_feature = hash_features
            return hash_table_feature
        
    def get_features(self):
        
        elf_feature = {}
        
        if self.file_check():          
            self.header_feature = self.elf_header()
            self.section_header_feature = self.section_header()
            self.program_header_feature = self.program_header()
            self.symbol_section_feature = self.symbol_table()
            self.dynamic_section_feature = self.dynamic_symbol()
            self.dynamic_symbol_section_feature = self.dynamic_symbol_section()
            self.global_offeset_table_feature = self.global_offset_table()
            self.hash_table_feature = self.hash_table()
            
            elf_feature.update(self.header_feature)
            elf_feature.update(self.section_header_feature)
            elf_feature.update(self.program_header_feature)
            elf_feature.update(self.symbol_section_feature)
            elf_feature.update(self.dynamic_section_feature)
            elf_feature.update(self.dynamic_symbol_section_feature)
            elf_feature.update(self.global_offeset_table_feature)
            elf_feature.update(self.hash_table_feature)
            
            ordered_values = [elf_feature[key] for key in self.feature_name_order]            
        else:
            print(f"elf check error: {self.binary_path}")
            ordered_values = [None for key in self.feature_name_order]
        
        return ordered_values