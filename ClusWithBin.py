class Cluster_w_Binaries(Cluster):
    """
    Cluster sub-class that produces a *resolved* stellar cluster with
    binary evolution accounted for.
    A table is output with the synthetic photometry and intrinsic
    properties of the individual stars (or stellar systems, if
    mutliplicity is used in the IMF object).

    A second table is produced that
    contains the properties of the companion stars independent of their
    primary stars.

    Parameters
    -----------
    iso: isochrone_binary object
        SPISEA binary_isochrone object
    imf: imf object
        SPISEA IMF object
    cluster_mass: float
        Total initial mass of the cluster, in M_sun
    ifmr: ifmr object or None
        If ifmr object is defined, will create compact remnants
        produced by the cluster at the given isochrone age. Otherwise,
        no compact remnants are produced.
    seed: int
        If set to non-None, all random sampling will be seeded with the
        specified seed, forcing identical output.
        Default None
    vebose: boolean
        True for verbose output.
    NOTE: the IMF MUST be such that multiples are made (i.e.
    muliplicity CANNOT BE NONE).
    Otherwise, why else would we want to use BPASS?

    ====================
    
    """
    def __init__(self, iso, imf, cluster_mass, ifmr=None, verbose=True,
                 seed=None):
        Cluster.__init__(self, iso, imf, cluster_mass,
                         ifmr=ifmr, verbose=verbose,
                         seed=seed)
        # Provide a user warning is random seed is set
        if seed is not None:
            print('WARNING: random seed set to %i' % seed)
        t1 = time.time()
        self.iso = iso
        self.verbose = verbose
        #####
        # Sample the IMF to build up our cluster mass.
        # NOTE: I have adjustment factor in place
        # in order to account for the amount of stars and star mass that
        # may be weeded out due to making initial masses come from the
        # isochrone
        #####

        mass, isMulti, compMass, sysMass = imf.generate_cluster(cluster_mass,seed=seed)
            
                
           
        # Figure out the filters we will make.
        # Names of the photometric filters as they would appear in a table
        self.filt_names = self.iso.filt_names_table
        # Below: INTENDED cluster mass!
        self.cluster_mass = cluster_mass
        # A little sanity check inserted to make sure that IMF generated
        # cluster mass is close to the desired cluster mass
        np_min_mass = np.min([sysMass.sum(), cluster_mass])
        #####
        # Make a table to contain all the information
        # about each stellar system.
        #####
        single_star_systems, self.non_matchable = self.make_singles_systems_table(isMulti, sysMass)
        #####
        # Make a table to contain all the information
        # about companions.
        #####
        if self.imf.make_multiples:
            # Temporary companion mass finder:
            self.intended_companions_mass = 0
            for lis in compMass:
                self.intended_companions_mass += sum(lis)
            self.intended_singles_mass = sysMass[np.where(~isMulti)[0]].sum()
            self.intended_primaries_mass = (sysMass[np.where(isMulti)[0]].sum() -
                                            self.intended_companions_mass)
            companions, double_systems = \
            self.make_primaries_and_companions(sysMass, compMass)
            self.star_systems = vstack([double_systems, single_star_systems])
            self.companions = companions
            
        else:
            self.star_systems = single_star_systems
            self.intended_primaries_mass = 0
            self.intended_companions_mass = 0
            self.intended_singles_mass = sysMass.sum()
        #####
        # Save our arrays to the object
        #####
        self.actual_cluster_mass = self.star_systems['systemMass'].sum()
        
        return

    def set_columns_of_table(self, t, N_systems, multi=False):
        """
        Input: t <--> Astropy Table with entries for initial masses
        of the stars.
        N_systems <--> Length of AstroPy Table
        Output/Result: adds columns for basic parameters to table t with their
        descriptions (e.g. metallicity or Teff)
        """
        t.add_column(Column(np.zeros(N_systems, dtype=float),
                            name='Teff'))
        t.add_column(Column(np.empty(N_systems, dtype=float),
                            name='L'))
        t.add_column(Column(np.empty(N_systems, dtype=float),
                            name='logg'))
        t.add_column(Column(np.repeat(False, N_systems),
                            name='isWR'))
        t.add_column(Column(np.empty(N_systems, dtype=float),
                            name='mass_current'))
        t.add_column(Column(np.empty(N_systems, dtype=int),
                            name='phase'))
        t.add_column(Column(np.repeat(False, N_systems),
                            name='touchedP'))
        t.add_column(Column(np.repeat(self.iso.metallicity,
                                      N_systems),
                            name='metallicity'))
        t.add_column(Column(np.repeat(multi, N_systems),
                                       name='isMultiple'))
        t.add_column(Column(np.repeat(False, N_systems),
                            name='merged'))
        # Add the filter columns to the table. They are empty so far.
        # Keep track of the filter names in : filt_names
        for filt in self.filt_names:
            t.add_column(Column(np.zeros(N_systems,
                                         dtype=float),
                                name=filt))
        if multi:
            t.add_column(Column(np.arange(N_systems),
                                name='designation'))
            
        return None

    def applying_IFMR_stars(self, stars,
                            comps=False, star_sys=None):
        """
        Input:
        stars: Astropy table of stars (either primary stars or companions)
        comps: whether the stars table is the table of companion stars.
        We apply the IFMR (if there's any given) to the
        a table of stars. Specifically, when stars have phase of 5 but
        have an effective temperature of 0, or have logg >= 6.9 or have
        non-physical temperatures (and are not merged with the primary)
        This is in order to make sure there are no compact stars masquerading
        as non-compact stars.
        """
        star_systems_phase_non_nan = np.nan_to_num(stars['phase'],
                                                   nan=-99)
        bad = np.where((star_systems_phase_non_nan > 5) &
                       (star_systems_phase_non_nan < 101) &
                       (star_systems_phase_non_nan != -99))
        # Print warning, if desired
        if self.verbose:
            for ii in range(len(bad[0])):
                print('WARNING: changing phase ' +
                      '{0} to 5'.format(star_systems_phase_non_nan[bad[0]
                                        [ii]]))
        stars['phase'][bad] = 5
        #####
        # Make Remnants for phase 5 stars with nan and for
        # stars with 0 Kelvin Teff
        # I decided to use the ifmr in a different context.
        #####
        
        cdx_rem = np.where((stars['Teff'] == 0) |
                           (~np.isfinite(stars['Teff'])) |
                           ((stars['logg'] >= 6.9) &
                            (stars['logg']==5)))[0]
        if comps:
            # A secondary star should be designated as a secondary
            # when it has high surface gravity (>= 6.9)
            # 0 Teff (Should not happen)
            # or if it has non-finite Teff but is not
            # merged with the primary.
            # The 6.9 Comes from the minimum mass set
            # in HOKI for a star to be considered a white dwarf
            # I assume that the surface gravities of neutron stars
            # and black holes are generally greater than those of
            # white dwarf stars
            cdx_rem = np.where((stars['phase'] == 5) & ((stars['Teff'] == 0) |
                               (stars['logg'] >= 6.9) |
                               ((~np.isfinite(stars["Teff"])) &
                                (~(stars['the_secondary_star?'] &
                                   (star_sys['merged']
                                    [stars['system_idx']]))))))[0]
        if self.ifmr:
            # Identify compact objects as those with Teff = 0.
            # Conditions the star has to be not merged and the star has to be
            # Calculate remnant mass and ID for compact
            # objects; update remnant_id and
            # remnant_mass arrays accordingly
            cond = ('metallicity_array' in
                    inspect.getfullargspec(self.ifmr.generate_death_mass).args)
            if cond:
                r_mass_tmp, r_id_tmp = \
                self.ifmr.generate_death_mass(mass_array=stars
                                              ['mass'][cdx_rem],
                                              metallicity_array=stars
                                              ['metallicity'][cdx_rem])
            else:
                r_mass_tmp, r_id_tmp = \
                self.ifmr.generate_death_mass(mass_array=stars
                                              ['mass'][cdx_rem])
            # Drop remnants where it is not relevant
            # (e.g. not a compact object or
            # outside mass range IFMR is defined for)
            good = np.where(r_id_tmp > 0)
            cdx_rem_good = cdx_rem[good]
            stars['mass_current'][cdx_rem_good] = r_mass_tmp[good]
            # when we have somee undefined or 0 temperature,
            # shouldn't we practically have no light coming
            # out of it (not even blackbody radiation)?
            stars['L'][cdx_rem] = 0.0
            stars['phase'][cdx_rem_good] = r_id_tmp[good]
            for filt in self.filt_names:
                stars[filt][cdx_rem] = \
                np.full(len(cdx_rem), np.nan)
        else:
            # If the IFMR doesn't exist, then we may need to
            # get rid of the stars that would warrant usage of the IFMR.
            stars.remove_rows(cdx_rem)
        return None
                
    def generate_2body_parameters(self, star_systemsPrime, companions):
        """
        Input:
        star_systemsPrime:Astropy table of the primary stars of the cluster
        companions: Astropy table of the companion stars of the cluster
        Generate the log_separation, e, i, omegas,
        (2-body problem parameters) using the given
        multiplicity.
        
        """

        N_comp_tot = len(companions)
        inst = isinstance(self.imf._multi_props,
                          multiplicity.MultiplicityResolvedDK)
        if inst:
            companions.add_column(Column(np.zeros(N_comp_tot, dtype=float),
                                         name='log_a'))
            companions.add_column(Column(np.zeros(N_comp_tot, dtype=float),
                                         name='e'))
            companions.add_column(Column(np.zeros(N_comp_tot, dtype=float),
                                         name='i', description='degrees'))
            companions.add_column(Column(np.zeros(N_comp_tot, dtype=float),
                                         name='Omega'))
            companions.add_column(Column(np.zeros(N_comp_tot, dtype=float),
                                         name='omega'))
            for ii in range(len(companions)):
                # If the mass of the star is less than 0.1 I will
                # then let the log_a be 0.
                # Do not want to trigger an error but I will try
                # NOT to cause and instead make
                # log_a = np.nan. That will be our indicator that
                # I could not calculate the log_a using
                # Duchene and Krauss and that we will
                # not use the log(a)
                try:
                    companions['log_a'][ii] = \
                    self.imf._multi_props.log_semimajoraxis(star_systemsPrime
                                                            ['mass']
                                                            [companions
                                                             ['system_idx']
                                                             [ii]])
                except ValueError:
                    # Indicator that we can't use the Multiplicity for log_a
                    companions['log_a'][ii] = np.nan
                    continue
            companions['e'] = \
            self.imf._multi_props.random_e(np.random.rand(N_comp_tot))
            props = self.imf._multi_props
            companions['i'], companions['Omega'], companions['omega'] = \
            props.random_keplarian_parameters(np.random.rand(N_comp_tot),
                                              np.random.rand(N_comp_tot),
                                              np.random.rand(N_comp_tot))
        else:
            # Indicator that we can't use the Multiplicity for log_a
            companions['log_a'] = np.nan
            companions['e'] = np.nan
            companions['i'], companions['Omega'], companions['omega'] = \
            np.nan, np.nan, np.nan
        return None
    
    def finding_secondary_stars(self, star_systemsPrime, companions):
        """
        Given the table so far of the primary stars and of the companions
        we try to find the secondary of the primary star.
        star_systemsPrime: table of primary stars (with companinos)
        companions: table of companions (currently including corresponding index
        Output: compMass_IDXs a hashmap similar in purpose to 
        """
        compMass_IDXs = {}
        max_log_gs = {}
        
        for x in range(len(star_systemsPrime)):
            # subtbl stands for sub-table. (of companions whose primary star
            # is at index x.
            subtbl = companions[np.where(companions['system_idx'] == x)[0]]
            subtbl2 = copy.deepcopy(subtbl)
            max_log_gs[x] = [0, 0]
            # handling nan'ed separations. We generally don't want to
            # consider for minimum distance and maximum gravitational
            # influence to the primary
            subtbl2['log_a'][np.where(~np.isfinite(subtbl['log_a']))] = np.inf
            max_log_gs[x][0] = np.max(subtbl2['mass']/(10 ** subtbl2['log_a']) ** 2)
            max_log_gs[x][1] = np.max(subtbl2['mass'])
              
            # Indicate where in the compMass array for star system
            # number x we are in
            # when we start looking at the companion
            compMass_IDXs[x] = 0
            # This is where the matching of the primary and the secondary begin
            # I will be tracking the number of bad star systems throguh the
            # variable rejected_system and rejected_companion
        return compMass_IDXs, max_log_gs

    def filling_in_primaries_and_companions(self, star_systemsPrime,
                                            companions, compMass_IDXs,
                                            min_log_gs, compMass):
        """
        Matches IMF generated systems to BPASS isochrone
        primary-secondary pairs  by closeness of primary stars
        - companion star - log separation tuples
        and fills tables using photometry, luminosity, and dynamics
        values from match in isochrone to describe star systems and
        secondary companions.

        Also, matches IMF generated tertiary companions with stars
        from BPASS isochrone's single stars by initial mass and fills
        companions table using photometry, luminosity, and dynamics values
        from match in isochrone to describe star systems'
        higher-order companions.
        
        Inputs: star_systems: Table containing information so far
        about IMF generated star systems and their primary stars 
        
        companions: Table containing information so far about IMF
        generated companions.
        
        Output: rejected_system, rejected_value: respectively,
        the number of star systems that could not be matched to a close
        enough isochrone counterpart 
        and the number of companions that could not be matched to one
        from the isochrone

        """
        rejected_system = 0
        rejected_companions = 0
        self.unmatched_tertiary = []
        self.unmatched_primary_pairs = []
        for x in range(len(companions)):
            sysID = companions[x]['system_idx']
            companions['mass'][x] = compMass[sysID][compMass_IDXs[sysID]]
            # First I find if the companion star has the greatest
            # possible gravitational pull on it.
            # if so, we've found the primary
            cond = np.isclose(companions['mass'][x]/(10 ** (-2 * companions['log_a'][x])),
                              min_log_gs[sysID][0], rtol=1e-2)
            if not min_log_gs[sysID][0]:
                cond = np.isnan(companions['log_a'][x]) and \
                np.isclose(companions['mass'][x], min_log_gs[sysID][1], atol=0.01)

            if cond:
                ind = match_binary_system(star_systemsPrime[sysID]['mass'],
                                          compMass[sysID]
                                          [compMass_IDXs[sysID]],
                                          companions['log_a'][x],
                                          self.iso,
                                          not np.isnan(companions['log_a']
                                                   [x]))
                ind = ind[np.where(ind != -1)[0]]
                if ((not len(ind)) or star_systemsPrime['bad_system'][sysID]):
                    self.unmatched_primary_pairs.append([star_systemsPrime[sysID]['mass'],
                                                         compMass[sysID]
                                                         [compMass_IDXs[sysID]]])
                    star_systemsPrime['bad_system'][sysID] = True
                    companions['bad_system'][np.where(companions['system_idx'] ==
                                                      sysID)] = True
                    compMass_IDXs[sysID] += 1
                    rejected_system += 1
                    rejected_companions += 1
                    continue

                ind = ind[0]
                star_systemsPrime['touchedP'][sysID] = True
                star_systemsPrime['Teff'][sysID] = \
                self.iso.primaries['Teff'][ind]
                star_systemsPrime['L'][sysID] = \
                self.iso.primaries['L'][ind]
                star_systemsPrime['logg'][sysID] = \
                self.iso.primaries['logg'][ind]
                star_systemsPrime['isWR'][sysID] = \
                np.round(self.iso.primaries['isWR'][ind])
                star_systemsPrime['mass'][sysID] = \
                self.iso.primaries['mass'][ind]
                star_systemsPrime['mass_current'][sysID] = \
                self.iso.primaries['mass_current'][ind]
                star_systemsPrime['phase'] = \
                np.round(self.iso.primaries['phase'][ind])
                if not (np.round(self.iso.primaries['phase'][ind])):
                    print("Bad phase")
                star_systemsPrime[sysID]['merged'] = \
                np.round(self.iso.secondaries['merged'][ind])
                table = self.iso.secondaries
                for filt in self.filt_names:
                    star_systemsPrime[sysID][filt] = \
                    self.iso.primaries[filt][ind]
                    companions[filt][x] = self.iso.secondaries[filt][ind]
            else:
                ind = match_model_uorder_companions(self.iso.singles['mass'],
                                                    np.array([compMass[sysID]
                                                              [compMass_IDXs[sysID]]]),
                                                    self.iso)
                ind = ind[np.where(ind != -1)[0]]
                if ((not len(ind)) or companions['bad_system'][x]):
                    # This means we cannot find a close enough companion.
                    companions['bad_system'][x] = True
                    self.unmatched_tertiary.append(compMass[sysID][compMass_IDXs
                                                                   [sysID]])
                    compMass_IDXs[sysID] += 1
                    rejected_companions += 1
                    continue
                ind = ind[0]
                table = self.iso.singles
                for filt in self.filt_names:
                    companions[filt][x] = table[filt][ind]
            # Obtain data on the  photometry of the companinons
            if (star_systemsPrime['merged'][sysID]):
                companions['log_a'][x] = np.nan 
            companions['Teff'][x] = table['Teff'][ind]
            companions['L'][x] = table['L'][ind]
            companions['logg'][x] = table['logg'][ind]
            companions['isWR'][x] = np.round(table['isWR'][ind])
            companions['mass'][x] = table['mass'][ind]
            companions['mass_current'][x] = table['mass_current'][ind]
            companions['phase'][x] = np.round(table['phase'][ind])
            # We want to look at the NEXT companion once we come back
            # to the system.
            compMass_IDXs[sysID] += 1
            # Obtain whether the companion is the secondary star
            # and not a tertiary or farther.
            companions['the_secondary_star?'][x] = cond
        self.unmatched_tertiary = np.array(self.unmatched_tertiary)
        self.unmatched_primary_pairs = np.array(self.unmatched_primary_pairs)
        return rejected_system, rejected_companions

    def adding_up_photometry(self, star_systemsPrime, companions):
        """ Adds up initial masses of secondaries to initial mass
        of the primary to create the systemMass values.
        Also adding up the photometries (kind of hard and ugly)
        to find each star systems' aggregate photometry.
        """

        for x in range(len(star_systemsPrime)):
            # Find all companions of the star system
            sub_tbl = companions[np.where(companions['system_idx'] == x)[0]]
            sum_of_comp = sub_tbl['mass'].sum()
            # Obtain the system mass.
            star_systemsPrime['systemMass'][x] = sum_of_comp + \
            star_systemsPrime['mass'][x]
            if (star_systemsPrime['phase'][x] == np.nan):
                star_systems_phase_non_nan = -99
            else:
                star_systems_phase_non_nan = star_systemsPrime['phase'][x]
            cond = ((int(star_systems_phase_non_nan) > 5) and
            (int(star_systems_phase_non_nan) < 101) and
            (int(star_systems_phase_non_nan) != -99))
            if (cond):
                print("Changing phase of primaries")
                if (self.verbose):
                    print('WARNING: changing phase {0} to 5'.format(star_systems_phase_non_nan))
                star_systemsPrime['phase'][x] = 5
            for filt in self.filt_names:
                # Magnitude of primary star (initially)
                mag_s = star_systemsPrime[filt][x]
                # Companion stars corresponding to the primary star
                comps = companions[np.where((companions['system_idx'] == x) &
                                            np.isfinite(companions[filt]))[0]]
                # trying to obtain as many finite magnitudes as I can
                if (not len(comps[filt])):
                    mag_c = np.nan
                else:
                    # Finding the sum of fluxes and then taking magnitude.
                    mag_c = comps[filt]
                # Add companion flux to system flux.
                f1 = 10 ** (-1 * mag_s / 2.5)
                f2 = 10 ** (-1 * mag_c / 2.5)

                f1 = np.nan_to_num(f1)
                f2 = np.nan_to_num(f2)
                f2 = np.sum(f2)
                # Good and bad systems
                
                # If *both* objects are dark, then keep the magnitude
                # as np.nan. Otherwise, add fluxes together
                if (f1 != 0 or f2 != 0):
                    star_systemsPrime[filt][x] = -2.5 * np.log10(f1 + f2)
                else:
                    star_systemsPrime[filt][x] = np.nan
                

    def make_singles_systems_table(self, isMulti, sysMass):
        """
        Make a part of the star_systems table and get synthetic photometry
        for each single star system. 

        Input: isMulti <--> Whether a star system is a
        multi-star system. (output of the IMF) 
        sysMass <--> System masses of all stellar systems (generated by IMF)

        Output: A part of the star_systems table that contains all
        information regarding single star systems.
        Data values are matched from rows of the self.isochrone
        based on the row's initial primary mass.
        """
        # We will be only looking for stars that are not multiple systems
        sysMass = sysMass[np.where(~ isMulti)[0]]
        old_sysMass = sysMass
        indices = \
        match_model_sin_bclus(np.array(self.iso.singles['mass']),
                                       sysMass, self.iso,
                              not self.imf.make_multiples)
        del_mass = sysMass[np.where(indices == -1)[0]].sum()
        # Notice: DELETED IS BEFORE IFMR APPLICATION
        deleted = len(indices[np.where(indices == -1)[0]])
        del_in = np.where(indices == -1)[0]
        indices = indices[np.where(indices != -1)[0]]
        N_systems = len(indices)
        sysMass = sysMass[indices]
        star_systems = Table([sysMass, sysMass],
                             names=['mass', 'systemMass'])
        self.set_columns_of_table(star_systems, N_systems)
        # Add columns for the Teff, L, logg, isWR,
        # mass_current, phase for the primary stars.
        star_systems['Teff'] = self.iso.singles['Teff'][indices]
        star_systems['L'] = self.iso.singles['L'][indices]
        star_systems['logg'] = self.iso.singles['logg'][indices]
        star_systems['isWR'] = self.iso.singles['isWR'][indices]
        star_systems['mass'] = self.iso.singles['mass'][indices]
        star_systems['systemMass'] = self.iso.singles['mass'][indices]
        star_systems['mass_current'] = \
        self.iso.singles['mass_current'][indices]
        star_systems['phase'] = self.iso.singles['phase'][indices]
        star_systems['metallicity'] = np.ones(N_systems) * \
        self.iso.metallicity
        self.applying_IFMR_stars(star_systems)
        for filt in self.filt_names:
            star_systems[filt] = self.iso.singles[filt][indices]
        print("{} single stars had to be deleted".format(deleted))
        print("{} solar masses".format(del_mass) +
              " had to be deleted from single stars before" +
              " application of the IFMR")
        star_systems.remove_columns(['touchedP'])
        return star_systems, old_sysMass[del_in]

    def make_primaries_and_companions(self, star_systems, compMass):
        """
        Input: star_systems--> numpy array of all star_system
        masses (generated by IMF)

        compMass--> list of lists of masses of companions of each star system

        Output:Creates tables star_systemsPrime and companions,
        which contain data regarding
        star systems with multiple stars
        and the companions.
        
        ====================
        Makes the primary stars and companions for non-single star systems of
        the star_systems table and get synthetic
        photometry for each single star system. Creates remnants when necessary.
        Given an initial primary star mass, initial companion mass, and log
        separation (last one is included if applicable) between the star and
        the companion, we designate the star with the secondary as the star with
        least separation from the primary. If there are multiple stars with the
        same value for separation, the star with the most mass is selected out of
        the set of companions with the same minimum separation and is designated
        as the secondary.
        
        Primaries and secondaries are matched to the pairs of stars with initial
        primary mass - initial secondary mass - log_current separation from the
        isochrone that are closest to the values generated by the IMF. Given that
        the error between the isochrone star system and the imf generated system
        is small enough, the star system is included in the cluster’s star_systems
        and companions tables.
        
        Tertiary and higher order companions are included, but are matched to single
        star models. As  similar to the policy with matching the primary-secondary pairs,
        if the companion deviates too much, in this case in terms of initial mass, from
        the most similar single star from the isochrone, it does not become part of
        the cluster object.
        """

        # Obtain the indices of the systems corresponding to each companion
        # make star_systems contain only multi-star systems
        indices = [x for x in range(len(compMass)) if len(compMass[x])]
        star_systems = star_systems[indices]
        # For each star system, the total mass of companions
        compMass_sum = np.array([sum(compMass[x]) for x in indices])
        compMass = np.array([compMass[x] for x in indices])
        # Make star_systems array only contain the masses of the primary stars
        star_systems = star_systems - compMass_sum
        # Number of multi-body star systems
        N_systems = len(star_systems)
        #####
        # MULTIPLICITY
        # Make a second table containing all the companion-star masses.
        # This table will be much longer... here are the arrays:
        # sysIndex - the index of the system this star belongs too
        # mass - the mass of this individual star.
        N_companions = np.array([len(star_masses) for star_masses in compMass])
        N_comp_tot = N_companions.sum()
        system_index = np.repeat(range(len(indices)), N_companions)
        # From now on, we will try to use the
        # new indices (i.e. what the 0-index system is
        # after we get rid of the single systems from the
        # star systems table) as much as possible.
        # Shows which star system number (index of primary star)
        # each companion star corresponds to
        companions = Table([system_index], names=['system_idx'])
        star_systemsPrime = Table([star_systems, np.zeros(N_systems,
                                                           dtype=float)],
                                  names=['mass', 'systemMass'])
        # Create a table for primary stars
        N_systems = len(indices)
        # Add columns for the Teff, L, logg, isWR,
        # mass_current, phase for the primary stars.
        self.set_columns_of_table(star_systemsPrime, N_systems, multi=True)
        
        # The following column indicates whether a star system
        # will have to be deleted due to lack of good match.
        star_systemsPrime.add_column(Column(np.repeat(False, N_systems),
                                            name='bad_system'))
        # Designation is really the index of the star,
        # but this may not be equal to
        # the 0, 1.., len -1 index iterable of the
        # cluster once we kill off the bad systems (systems whose
        # primary-secondary pair could not be found
        
        # Add columns for the Teff, L, logg, isWR mass_current,
        # phase, and filters for the companion stars.
        companions.add_column(Column(np.zeros(N_comp_tot, dtype=float),
                                     name='mass'))
        self.set_columns_of_table(companions, N_comp_tot)
        
        # Marks whether the system was unmatchable and
        # its row must be deleted.
        companions.add_column(Column(np.repeat(False, N_comp_tot),
                                     name='bad_system'))
        companions.add_column(Column(np.repeat(False, N_comp_tot),
                                     name='the_secondary_star?'))
        for ind in range(len(star_systemsPrime)):
            (companions['mass'][np.where
                                (companions['system_idx'] ==
                                 ind)]) = compMass[ind]
        
        self.generate_2body_parameters(star_systemsPrime, companions)
        # What-th companion of the system am I inspecting
        # Index of the companion we are inspecting in the compMass
        # 'Tuple' of minimum log_a, initial mass of star.
        # i.e. initial log_a, mass of secondary star of a system
        compMass_IDXs, max_log_gs = (self.finding_secondary_stars(star_systemsPrime,
                                                                  companions))
        
        
        rejected_system, rejected_companions = \
        self.filling_in_primaries_and_companions(star_systemsPrime,
                                                 companions, compMass_IDXs,
                                                 max_log_gs, compMass)
        # =============
        # Now I delete the primary and companion which could
        # not be matched to a close-enough star in the isochrone
        # Get rid of the Bad_systems (un-matchable systems) and bad stars.
        # =============
        star_systemsPrime = (star_systemsPrime
                             [np.where(((~star_systemsPrime['bad_system']) | (~star_systemsPrime['touchedP'])))
                              [0]])
        companions = companions[np.where(~companions['bad_system'])[0]]
        # =============
        # Make the indices/designations of the star_systemsPrime
        # 0-indexed again.
        # Do some matching for the companions  so that
        # the system_idx tells us
        # that the companion matches to the system-idx-th
        # entry of the primary star table.
        # =============
        
        for x in range(len(companions)):
            companions['system_idx'][x] = \
            np.where(star_systemsPrime['designation'] ==
                     companions['system_idx'][x])[0][0]
        
        
        #####
        # Make Remnants for non-merged stars with nan
        # and for stars with 0 Kelvin Teff
        #####
        # Identify compact objects as those with Teff = 0 or 
        # the secondary stars that are non-merged and have a non-
        # finite temperature
        self.applying_IFMR_stars(companions, comps=True,
                                 star_sys=star_systemsPrime)
        self.applying_IFMR_stars(star_systemsPrime)
        # The compact remnants have photometric
        # fluxes of nan now. So now we can procede
        # with the photometry.
        self.adding_up_photometry(star_systemsPrime, companions)
        companions_phase_non_nan = np.nan_to_num(companions['phase'],
                                                 nan=-99)
        bad = np.where((companions_phase_non_nan > 5) &
                       (companions_phase_non_nan < 101) &
                       (companions_phase_non_nan != -99))
        if self.verbose:
            print("Running the changing phase on companions")
            for ii in range(len(bad[0])):
                print('WARNING: changing phase {0} to 5'.format(companions_phase_non_nan[bad[0][ii]]))
            companions['phase'][bad] = 5
        # Get rid of the columns designation and and bad_system
        star_systemsPrime.remove_columns(['bad_system', 'designation', 'touchedP'])
        companions.remove_columns(['bad_system', 'the_secondary_star?', 'touchedP'])
        if self.verbose:
            print("{} non-single star systems".format(str(rejected_system)) + 
                  " had to be deleted" +
                  " before IFMR application")
            print("{} companions".format(str(rejected_companions)) +
                  " had to be deleted before" +
                  " IFMR was applied")
        # For testing purposes we can make rejected_system and rejected_comapnions
        # be returned too.
        return companions, star_systemsPrime
