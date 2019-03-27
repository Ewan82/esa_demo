program runsim
  use mo_sensimul
  implicit none

  ! commandline controlled
  logical :: s1_no_vh      = .false.
  logical :: s1a_no_vh     = .false.
  logical :: s1b_no_vh     = .false.
  logical :: s1_no_vv      = .false.
  logical :: s1a_no_vv     = .false.
  logical :: s1b_no_vv     = .false.
  logical :: s2_no_visnir  = .false.
  logical :: s2_no_swnir   = .false.
  logical :: s2a_no_visnir = .false.
  logical :: s2a_no_swnir  = .false.
  logical :: s2b_no_visnir = .false.
  logical :: s2b_no_swnir  = .false.

  ! other
  integer             :: j, n, m
  real(kind=8), allocatable :: x(:), sx(:), ysim(:), xphys(:)
  logical :: exist
  logical :: ldebug = .true.

  !-------------------
  ! command line
  integer :: narg !#of arg & counter of arg
  integer :: nskip
  character(len=64) :: argname,option
  character(len=64) :: argval


  !-------------------
  !     command line
  !
  narg=command_argument_count()
  !Loop over the arguments
  if(narg>0)then
     !loop across options
     nskip=0
     arg: do j=1,narg
        if( nskip.gt.0 ) then
           nskip = nskip - 1
           cycle arg
        endif
        call get_command_argument(j,argname)
        select case(adjustl(argname))
        case("--help","-h")
           call options_help()
           stop 0
           exit arg
        case('--s1_no_vh')
           s1_no_vh = .true.
        case('--s1a_no_vh')
           s1a_no_vh = .true.
        case('--s1b_no_vh')
           s1b_no_vh = .true.
        case('--s1_no_vv')
           s1_no_vv = .true.
        case('--s1a_no_vv')
           s1a_no_vv = .true.
        case('--s1b_no_vv')
           s1b_no_vv = .true.
        case('--s2_no_visnir')
           s2_no_visnir = .true.
        case('--s2_no_swnir')
           s2_no_swnir = .true.
        case('--s2a_no_visnir')
           s2a_no_visnir = .true.
        case('--s2a_no_swnir')
           s2a_no_swnir = .true.
        case('--s2b_no_visnir')
           s2b_no_visnir = .true.
        case('--s2b_no_swnir')
           s2b_no_swnir = .true.
        case default
           write(*,'(a)') "ERROR::Option ",adjustl(argname),"unknown"
           stop 0
        end select
     end do arg
  end if


  !-------------------
  !-- init model
  write(*, '(a)') ' INFO::runsim:calling initf...'
  call initf(n, m)
  write(*, '(a)') ' INFO::runsim:...done.'
  if( ldebug ) then
     write(*, '(a,2(a,i3,1x))') ' DEBUG::runsim:initf yields:',&
          'n=',n,'m=',m
  endif


  !-------------------
  ! allocate arrays
  allocate(x(n),xphys(n),sx(n),ysim(m))

  !-------------------
  ! init unknowns !!!physical values!!!
  write(*, '(a)') ' INFO::runsim:calling initx...'
  call initx(n, x, sx)
  write(*, '(a)') ' INFO::runsim:...DONE.'

  !-------------------
  ! possibly overwrite
  !
  inquire(FILE='x-external.b',EXIST=exist)
  if (exist) then
     open (unit=1, file='x-external.b', form='unformatted')
     read (1) x
     close (1)
     write(*, '(a)') ' INFO::runsim:have read control vector from x-external.b'//&
          ' !!!expecting scaled control vector!!!'
  endif


  !--
  call x2p(n, x, xphys)

  if( ldebug ) then
     write(*,'(a)') ' DEBUG::runsim::calling simulate_s1s2 at x ...'
     write(*, '(a3,3(a15))' ) 'j', 'x-physical', 'x-scaled', 'x-sigma'
     do j=1,n
        write(*, '(i3,3(f15.8))' ) j, xphys(j), x(j), sx(j)
     enddo
  endif

  !-- run combined S1+S2 simulation
  write(*,'(a)') ' INFO::runsim:calling combinded S1+S2 simulation with simulate_s1s2...'
  call simulate_s1s2(n, x, m, ysim)
  write(*,'(a)') ' INFO::runsim:...done.'

  !-- apply filtering (if selected by user)
  !-- S1 failure scenarios
  if( s1_no_vh ) then
     s1_failure_satellite = 'S1'
     s1_failure_pol       = 'vh'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S1A/S1B without VH"'
  else if( s1a_no_vh ) then
     s1_failure_satellite = 'S1A'
     s1_failure_pol       = 'vh'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S1A without VH"'
  else if( s1B_no_vh ) then
     s1_failure_satellite = 'S1B'
     s1_failure_pol       = 'vh'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S1B without VH"'
  else if( s1_no_vv ) then
     s1_failure_satellite = 'S1'
     s1_failure_pol       = 'vv'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S1A/S1B without VV"'
  else if( s1a_no_vv ) then
     s1_failure_satellite = 'S1A'
     s1_failure_pol       = 'vv'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S1A without VV"'
  else if( s1B_no_vv ) then
     s1_failure_satellite = 'S1B'
     s1_failure_pol       = 'vv'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S1B without VV"'
  endif
  !-- S2 failure scenarios
  if( s2_no_visnir ) then
     s2_failure_satellite = 'S2'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S2A/S2B without vis/nir bands (1-9)"'
     s2_failure_bands(1:10) = .true.
  else if( s2_no_swnir ) then
     s2_failure_satellite = 'S2'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S2A/S2B without swnir bands (10-12)"'
     s2_failure_bands(11:13) = .true.
  else if( s2a_no_visnir ) then
     s2_failure_satellite = 'S2A'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S2A without vis/nir bands (1-9)"'
     s2_failure_bands(1:10) = .true.
  else if( s2a_no_swnir ) then
     s2_failure_satellite = 'S2A'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S2B without swnir bands (10-12)"'
     s2_failure_bands(11:13) = .true.
  else if( s2b_no_visnir ) then
     s2_failure_satellite = 'S2B'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S2B without vis/nir bands (1-9)"'
     s2_failure_bands(1:10) = .true.
  else if( s2b_no_swnir ) then
     s2_failure_satellite = 'S2B'
     write(*, '(a)')' INFO::runsim:apply failure scenario "S2B without swnir bands (10-12)"'
     s2_failure_bands(11:13) = .true.
  endif

  !-- 
  call simulation_failure_filter(m, ysim)

  !-- create simulation output
  call finishf(n, x, m, ysim)



  ! dispose memory
  call dispose()

  contains

    subroutine dispose()
      implicit none
      if( allocated(x) )    deallocate(x)
      if( allocated(xphys) )    deallocate(xphys)
      if( allocated(sx) )   deallocate(sx)
      if( allocated(ysim) ) deallocate(ysim)
    end subroutine dispose


    subroutine options_help()
      implicit none
      write(*, '(/)')
      write(*, '(a)') '===================='
      write(*, '(a)') ' available command line options:'
      argname = '--s1_no_vh'
      write(*, fmt=3333) argname, 'VH simulation at S1A/S1B acquisition times set to fill-value'
      argname = '--s1a_no_vh'
      write(*, fmt=3333) argname, 'VH simulation at S1A acquisition times set to fill-value'
      argname = '--s1b_no_vh'
      write(*, fmt=3333) argname, 'VH simulation at S1B acquisition times set to fill-value'
      argname = '--s1_no_vv'
      write(*, fmt=3333) argname, 'Vv simulation at S1A/S1B acquisition times set to fill-value'
      argname = '--s1a_no_vv'
      write(*, fmt=3333) argname, 'VV simulation at S1A acquisition times set to fill-value'
      argname = '--s1b_no_vv'
      write(*, fmt=3333) argname, 'VV simulation at S1B acquisition times set to fill-value'
      argname = '--s2_no_visnir'
      write(*, fmt=3333) argname, 'BRF simulation at S2A/S2B acquisition times set to fill-value (bands 1-9)'
      argname = '--s2_no_swnir'
      write(*, fmt=3333) argname, 'BRF simulation at S2A/S2B acquisition times set to fill-value (bands 10-12)'
      argname = '--s2a_no_visnir'
      write(*, fmt=3333) argname, 'BRF simulation at S2A acquisition times set to fill-value (bands 1-9)'
      argname = '--s2a_no_swnir'
      write(*, fmt=3333) argname, 'BRF simulation at S2A acquisition times set to fill-value (bands 10-12)'
      argname = '--s2b_no_visnir'
      write(*, fmt=3333) argname, 'BRF simulation at S2B acquisition times set to fill-value (bands 1-9)'
      argname = '--s2b_no_swnir'
      write(*, fmt=3333) argname, 'BRF simulation at S2B acquisition times set to fill-value (bands 10-12)'
      write(*, '(a)') '===================='
      write(*, '(/)')
      3333 FORMAT (2x, a15,2x,a)
    end subroutine options_help
end program runsim
