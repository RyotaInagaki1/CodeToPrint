class Isochrone_Binary(Isochrone):
    """
    Base Isochrone class.

    Parameters
    ----------
    logAge : float
        The age of the isochrone, in log(years)
    AKs : float
        The total extinction in Ks filter, in magnitudes
    distance : float
        The distance of the isochrone, in pc
    metallicity : float, optional
        The metallicity of the isochrone, in [M/H].
        Default is 0.
    evo_model: model evolution class, optional
        Set the stellar evolution model class.
        Default is evolution.MISTv1().
    atm_func: model atmosphere function, optional
        Set the stellar atmosphere models for the stars.
        Default is get_merged_atmosphere.
    wd_atm_func: white dwarf model atmosphere function, optional
        Set the stellar atmosphere models for the white dwafs.
        Default is get_wd_atmosphere
    mass_sampling : int, optional
        Sample the raw isochrone every `mass_sampling` steps. The default
        is mass_sampling = 0, which is the native isochrone mass sampling
        of the evolution model.
    wave_range : list, optional
        length=2 list with the wavelength min/max of the final spectra.
        Units are Angstroms. Default is [3000, 52000].
    min_mass : float or None, optional
        If float, defines the minimum mass in the isochrone.
        Unit is solar masses. Default is None
    max_mass : float or None, optional
        If float, defines the maxmimum mass in the isochrone.
        Units is solar masses. Default is None.
    rebin : boolean, optional
        If true, rebins the atmospheres so that they are the same
        resolution as the Castelli+04 atmospheres. Default is False,
        which is often sufficient synthetic photometry in most cases.
    Important Attributes:
    Primaries-- AstroPy Table containing the following columns:
    -----------------------------------------------------------
    mass = in units of Solar Masses. Contains the initial mass of the primary
    star
    log_a = the log10(separation of the star with the secondary in AU)
    mass_current = in units of solar masses. Contains the initial mass of
    the primary star
    L = Luminosity of the primary star in Watts
    Teff -- in Kelvin. Effective temperature of the primary star
    R -- in units of solar radii. Radius of the primary star.
    phase -- integer indicating whether a primary star is a white dwarf
    (101) or not (12)
    gravity -- log10(acceleration of gravity at the primary star's surface
    in m/s^2)
    isWR -- boolean indicating whether a primary star is a WR Star
    ------------------------------------------------------------
    singles-- AstroPy Table containing the following columns:
    -----------------------------------------------------------
    mass = in units of Solar Masses. Contains the initial mass
    of the single star
    mass_current = in units of solar masses. Contains the
    initial mass of the single star
    L = Luminosity of the single star in Watts
    Teff -- in Kelvin. Effective temperature of the single star
    R -- in units of solar radii. Radius of the single star.
    phase -- integer indicating whether a single star is
    a white dwarf (101) or not (12)
    gravity -- log10(acceleration of gravity at the single
    star's surface in m/s^2)
    isWR -- boolean indicating whether a single star is a WR Star
    ------------------------------------------------------------
    secondaries -- AstroPy Table containing the following columns:
    -----------------------------------------------------------
    mass = in units of Solar Masses. Contains the initial mass
    of the secondary star
    mass_current = in units of solar masses. Contains the initial
    mass of the secondary star
    L = Luminosity of the secondary star in Watts
    Teff -- in Kelvin. Effective temperature of the single star
    R -- in units of solar radii. Radius of the secondary star.
    phase -- integer indicating whether a secondary star is a
    white dwarf (101) or not (12)
    logg -- log10(acceleration of gravity at the
    secondary star's surface in m/s^2)
    isWR -- boolean indicating whether a secondary star is a WR Star
    merged -- whether the corresponding primary star (WHICH IS AT THE
    SAME INDEX in the primaries table as is the secondary star),
    has actually incorporated the secondary.
    If merged is true, the L, T_eff, gravity, mass_current will be set
    to np.nan along with any magnitudes.
    ------------------------------------------------------------
    """

    def __init__(self, logAge, AKs, distance, metallicity,
                 evo_model=evolution.BPASS(), atm_func=default_atm_func,
                 wd_atm_func=default_wd_atm_func, mass_sampling=1,
                 red_law=default_red_law,
                 wave_range=[3000, 52000], min_mass=None, max_mass=None,
                 filters=['ubv,U', 'ubv,V', 'ubv,B', 'ubv,R', 'ubv,I'],
                 rebin=True):
        t1=time.time()
        self.metallicity = metallicity
        self.logage = logAge
        # Accounting for the definition of metallicity of the
        # evolution object's
        # Isochrone function.
        # Changes by Ryota: make atm_func and wd_atm_func instance vars
        self.atm_func = atm_func
        self.wd_atm_func = wd_atm_func
        self.distance = distance
        self.wave_range = wave_range
        self.AKs = AKs
        self.red_law = red_law
        self.filters = filters
        self.filt_names_table = []
        self.verbose = True

        # Assert that the wavelength ranges are within the limits of the
        # VEGA model (0.1 - 10 microns)
        try:
            assert wave_range[0] > 1000
            assert wave_range[1] < 100000
        except AssertionError:
            print('Desired wavelength range invalid. Limit to 1000 - 10000 A')
            return
        # Get solar metallicity models for a population at a specific age.
        # Takes about 0.1 seconds.
        evol = evo_model.isochrone(age=10 ** logAge,
                                   metallicity=metallicity)

        # Eliminate cases where log g is less than 0
        idx = np.where(evol['logg'] > 0)
        evol = evol[idx]

        # Trim to desired mass range
        if min_mass:
            idx = np.where(evol['mass'] >= min_mass)
            evol = evol[idx]
        if max_mass:
            idx = np.where(evol['mass'] <= max_mass)
            evol = evol[idx]

        # Give luminosity, temperature, mass, radius units (astropy units).
        L_all = 10 ** evol['logL'] * constants.L_sun  # luminsoity in W
        T_all = 10 ** evol['logT'] * units.K
        # TO DO: Conditionals to make sure I am using valid values

        R_all = np.sqrt(L_all / (4.0 * math.pi * c.sigma_sb * T_all ** 4))
        # masses in solar masses
        mass_all = evol['mass'] * units.Msun
        logg_all = evol['logg']
        mass_curr_all = evol['mass_current'] * units.Msun
        # We will keep track of the phase of the BPASS primary/
        # single star system
        phase_all = evol['phase']
        # We will keep track of the phase of the BPASS secondary
        phase_all2 = evol['phase2']
        # We will keep track of whether the star is WR or not
        isWR_all = evol['isWR']
        isWR2 = evol['isWR2']
        # Actual temperature of secondary in Kelvins
        Tef2 = (10 ** evol['log(T2)']) * units.K
        R2 = (10 ** evol['log(R2)']) * constants.R_sun
        L2 = (10 ** evol['log(L2)']) * constants.L_sun
        singles = Table([mass_all, L_all, T_all, R_all, logg_all, isWR_all,
                        mass_curr_all, phase_all, evol['single'], evol['source']],
                        names=['mass', 'L', 'Teff', 'R', 'logg',
                               'isWR', 'mass_current', 'phase', 'single', 'source'])
        # Also have inserted conversion factor to deal with the units of
        # log(a)
        primaries = Table([mass_all,
                           L_all, T_all, R_all, logg_all, isWR_all,
                           mass_curr_all, phase_all, evol['single'], evol['source']],
                          names=['mass', 'L', 'Teff', 'R', 'logg',
                                 'isWR', 'mass_current', 'phase',  'single', 'source'])
        # Note that we keep information regarding whether
        # a star corresponds to a merger.
        # This will help us decide whether we should not account
        # for the L2, Tef_2 during atmosphere creation
        # Yes I do unit conversions for log(a) column
        # as that is in log(a/R_sun)
        secondaries = Table([evol['mass2'] * units.Msun,
                            evol['log(a)'] + np.log10(constants.R_sun/constants.au),
                            L2, Tef2, R2,
                            evol['logg2'], isWR2,
                            evol['mass_current2'] * units.Msun,
                            phase_all2,
                            evol['single'],
                            evol['mergered?'], evol['source']],
                            names=['mass', 'log_a', 'L', 'Teff',
                                   'R', 'logg', 'isWR', 'mass_current',
                                   'phase', 'single', 'merged', 'source'])
        # Make sure that we are only looking at stars with companions when
        # examining the secondaries and primaries.
        secondaries = secondaries[np.where(~secondaries['single'])[0]]
        secondaries.remove_column('single')
        singles = singles[np.where(singles['single'])[0]]
        singles.remove_column('single')
        primaries = primaries[np.where(~primaries['single'])[0]]
        primaries.remove_column('single')
        # Trim down the table by selecting every Nth point where
        # N = mass sampling factor.
        singles = singles[::mass_sampling]
        primaries = primaries[::mass_sampling]
        secondaries = secondaries[::mass_sampling]
        # Inserting before I forget
        # I try to make sure that we have a queue
        # of atm_function results to process
        # If we have null values
        self.spec_list_si = [None for x in range(len(singles))]  # For single Stars
        self.spec_list2_pri = [None for x in range(len(primaries))]  # For primary stars
        self.spec_list3_sec = [None for x in range(len(secondaries))]  # For secondary stars
        # Turns into an attribute since we will access this in another function
        self.pairings2 = {"Singles": self.spec_list_si,
                          "Primaries": self.spec_list2_pri,
                          "Secondaries": self.spec_list3_sec}
        pairings = {"Singles": singles, "Primaries": primaries,
                    "Secondaries": secondaries}
        codes = {"Singles": 0, "Primaries": 1,
                    "Secondaries": 2}
        self.codes2 = {0: singles, 1: primaries,
                       2: secondaries}
        self.pairings = pairings
        self.singles = singles
        self.primaries = primaries
        self.secondaries = secondaries
        # For each temperature extract the synthetic photometry.
        for x in pairings:
            tab = pairings[x]
            atm_list = self.pairings2[x]
            # Workaround for a glitch encountered with units not showing up.
            # may need to come back and get rid of it since it looks silly.
            R_all = tab['R'] * units.m/units.m
            gravity_table = tab['logg']
            # a little issue with the compact remnant primaries from
            # the secondary star
            if x == 'Secondaries':
                merged = np.where(tab['merged'])
                tab['mass_current'][merged] = np.nan
                tab['Teff'][merged] = np.nan
                tab['R'][merged] = np.nan
                tab['logg'][merged] = np.nan
                tab['isWR'][merged] = False
                tab['phase'][merged] = -99
                                 
            cond = np.where(np.isfinite(tab['logg']) & (tab['logg'] != 0.0) &
                            np.isfinite(tab['L']) & (tab['L'] > 0.0) &
                            np.isfinite(tab['Teff']) & (tab['Teff'] > 0.0) &
                            np.isfinite(tab['R']) & (tab['R']> 0.0) &
                            (tab['phase'] <= 101) & (tab['phase'] != -99))[0]
            
            wrapper = BypasserContainer()
            wrapper.wave_range = wave_range
            wrapper.distance = distance
            wrapper.lis = atm_list
            vectorized_atm_maker = np.vectorize(self.atm_generator_to_vectorize)
            vectorized_atm_maker(cond, wrapper, wrapper, metallicity, atm_func,
                                 wd_atm_func, red_law, AKs, rebin, codes[x])
            bad_starcond = np.where(~ (np.isfinite(tab['logg']) &
                                       (tab['logg'] != 0.0) &
                                       np.isfinite(tab['L']) &
                                       (tab['L'] > 0.0) &
                                       np.isfinite(tab['Teff']) &
                                       (tab['Teff'] > 0.0) &
                                       np.isfinite(tab['R']) & (tab['R'] > 0.0) &
                                       (tab['phase'] <= 101) &
                                       (tab['phase'] != -99)))[0]
            tab['Teff'][bad_starcond] = np.nan
            tab['L'][bad_starcond] = np.nan
            tab['R'][bad_starcond] = np.nan
            tab['logg'][bad_starcond] = np.nan
        # I hope to change the next few lines to this as this will make
        # more thorough use of numpy and decrease the number of for-loop
        # iterations that are used.
                      
        self.singles = singles
        self.primaries = primaries
        self.secondaries = secondaries
        # Append all the meta data to the summary table.
        for tab in (singles, primaries, secondaries):
            tab.meta['REDLAW'] = red_law.name
            tab.meta['ATMFUNC'] = atm_func.__name__
            tab.meta['EVOMODEL'] = 'BPASS v2.2'
            tab.meta['LOGAGE'] = logAge
            tab.meta['AKS'] = AKs
            tab.meta['DISTANCE'] = distance
            tab.meta['METAL_IN'] = evol.meta['metallicity_in']
            tab.meta['METAL_ACT'] = evol.meta['metallicity_act']
            tab.meta['WAVEMIN'] = wave_range[0]
            tab.meta['WAVEMAX'] = wave_range[1]
        self.make_photometry()
        t2 = time.time()
        print('Isochrone generation took {0:f} s.'.format(t2-t1))
        return

    def atm_generator_to_vectorize(self, c_ind, w_wrapper, list_wrapper, met,
                                   atm_func, wd_atm_func, red_law, AKs, rebin,
                                   code):
        tab = self.codes2[code]
        wave_range = w_wrapper.wave_range
        distance = w_wrapper.distance
        atm_list = list_wrapper.lis
        R_all = tab['R'] * units.m / units.m
        R = float( R_all[c_ind].to('pc') / units.pc) 

        if (tab[c_ind]['phase'] < 101):
            star =  atm_func(temperature=tab['Teff'][c_ind],
                             gravity=tab['logg'][c_ind],
                             metallicity=met,
                             rebin=rebin)
        else:
            star =  wd_atm_func(temperature=tab['Teff'][c_ind],
                                gravity=tab['logg'][c_ind],
                                metallicity=met,
                                verbose=False)
        # Trim wavelength range down to
        # JHKL range (0.5 - 5.2 microns)
        star = spectrum.trimSpectrum(star, wave_range[0],
                                             wave_range[1])
        # Convert into flux observed at Earth (unreddened)
        R = float(R_all[c_ind].to("pc") / units.pc)
        star *= (R / distance) ** 2 # in erg s^-1 cm^-2 A^-1
        # Redden the spectrum. This doesn't take much time at all.
        red = red_law.reddening(AKs).resample(star.wave)
        star *= red
        # Save the final spectrum to our spec_list for later use.
        atm_list[c_ind] = star
        return
    
    def make_photometry(self, rebin=True, vega=vega):
        """
        Make synthetic photometry for the specified filters. This function
        udpates the self.points table to include new columns with the
        photometry.
        """
        startTime = time.time()

        meta = self.singles.meta

        print('Making photometry for isochrone: log(t) = %.2f  AKs = %.2f  dist = %d' %(meta['LOGAGE'], meta['AKS'], meta['DISTANCE']))
        print('     Starting at: ', datetime.datetime.now(),
              '  Usually takes ~5 minutes')
        # npoints = len(self.points)
        verbose_fmt = 'M = {0:7.3f} Msun  T = {1:5.0f} K  m_{2:s} = {3:4.2f}'

        # Loop through the filters, get filter info, make photometry for
        # all stars in this filter.
        for ii in self.filters:
            prt_fmt = 'Starting filter: {0:s}   Elapsed time: {1:.2f} seconds'
            print(prt_fmt.format(ii, time.time() - startTime))
            filt = get_filter_info(ii, rebin=rebin, vega=vega)
            filt_name = get_filter_col_name(ii)
            # Make the column to hold magnitudes
            # in this filter. Add to points table.
            col_name = 'm_' + filt_name
            self.filt_names_table.append(col_name)
            mag_col = Column(np.zeros(len(self.singles), dtype=float),
                             name=col_name)
            self.singles.add_column(mag_col)
            mag_col = Column(np.zeros(len(self.secondaries), dtype=float),
                             name=col_name)
            self.secondaries.add_column(mag_col)
            mag_col = Column(np.zeros(len(self.primaries), dtype=float),
                             name=col_name)
            self.primaries.add_column(mag_col)
            # Loop through each star in the isochrone and
            # do the filter integration
            print('Starting synthetic photometry')
            for x in self.pairings:
                listofStars = self.pairings2[x]
                table = self.pairings[x]
                length_of_list = len(listofStars)
                for ss in range(length_of_list):
                    star = listofStars[ss]
                    # Nada used to be a fill in value.
                    # I thought that None would be more intuitive
                    # So we will be using None
                    if star:
                        # These are already extincted, observed spectra.
                        star_mag = mag_in_filter(star, filt)
                    else:
                        # We want nan to stand in place
                        # for stars that do not have valid T
                        # or radius or L.
                        star_mag = np.nan
                    table[col_name][ss] = star_mag
                    if (self.verbose and (ss % 100) == 0):
                        print(verbose_fmt.format(table['mass'][ss],
                                                 table['Teff'][ss],
                                                 filt_name, star_mag))
        endTime = time.time()
        print('      Time taken: {0:.2f} seconds'.format(endTime - startTime))
        return
