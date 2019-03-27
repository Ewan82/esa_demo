program runua
  use mo_sensimul, only:finishdf
  implicit none
  integer :: n, mf, mjac
  integer :: ierr
  real(kind=8), allocatable :: x(:), sx(:), cx(:,:)
  integer :: iostat
  logical :: exist, tstflg
  logical :: ldebug = .false., diag = .true.
  logical :: run_targets !-- whether to apply target retrieval(s)

  ! commandline
  integer :: iarg,narg, nconsumed
  character(len=32) :: argval, argname, option

  !-- initialise
  run_targets = .true.
  ldebug = .false.
  diag = .true.

  !-- command line handling
  narg=command_argument_count()
  if(narg>0)then
     !loop across options
     nconsumed=0
     arg: do iarg=1,narg
        if( nconsumed.gt.0 ) then
           nconsumed = nconsumed - 1
           cycle arg
        endif
        call get_command_argument(iarg,argname)
        select case(adjustl(argname))
        case("--help","-h")
           write(*,'(/)')
           write(*, '(a)') '===================='
           write(*, '(1x,a)') 'available options:'
           option = '--no_targets'
           write(*,fmt=3333) option, 'disable target retrievals '//&
                '(Note: In particular, FAPAR retrieval may be time-consuming)'
           write(*, '(a)') '===================='
           write(*,'(/)')
           stop 0
           exit arg
        case('--no_targets')
           run_targets = .false.
        case default
           write(*,'(a)') "ERROR::Option ",adjustl(argname),"unknown"
           stop 0
        end select
     end do arg
  end if

  !-------------------
  !-- initialise retrieval procedure
  !
  call retrieval_read_ctl()
  call retrieval_dump_ctl()

  !-------------------
  !-- init model
  write(*, '(a)') ' INFO::runua:calling initf...'
  call initf(n, mf)
  mjac = mf + 2*n
  write(*, '(a)') ' INFO::runua:...done.'
  if( ldebug ) then
     write(*, '(a,6(a,i3,1x))') ' DEBUG::runua:initf yields:',&
          'n=',n,'mf=',mf
  endif

  !-------------------
  ! init unknowns
  allocate (x(n),sx(n),cx(n,n))
  write(*, '(a)') ' INFO::runua:calling initx...'
  call initx(n, x, sx)
  write(*, '(a)') ' INFO::runua:...done.'

  !-- control vector might be read from external file
  !   resulting from a retrieval procedure
  inquire(FILE='x.b',EXIST=exist)
  if(exist) then
     write(*, '(a)') ' INFO::runua:reading control vector from file ***'//'x.b'//'***'
     open (unit=1, file='x.b', form='unformatted', iostat=iostat)
     if( iostat.ne.0 ) then
        write(*, '(a)') ' FATAL::runua:'//&
             'failed opening existing file ***x.b***'
        stop
     endif
     read(1, iostat=iostat) x
     if( iostat.ne.0 ) then
        write(*, '(a)') ' FATAL::runua:'//&
             'failed reading control vector from file ***x.b***'
        stop
     endif
     close (1)
     write(*,'(a)') ' INFO::runua:...done'
  endif

  !-- init dynamic state-vector model
  call optim_is_state_term_enabled(tstflg)
  if( tstflg ) then
     call state_model_set()
  endif

  !-- uncertainty analysis
  call ua(mjac, n, x, cx, ldebug, diag, ierr)

  !-- logging/output
  call wsigma(6,n,sx,cx)
  call wsigma(1,n,sx,cx)

  !-- post processing (control vector components)
  call finishdf(n, x, cx)

  !-- informational output
  call printua(n, cx, ldebug)

  !-- target operator: HH-backscatter,FAPAR
  if( run_targets ) then
     write(*, '(a)') ' INFO::runua:'//&
          'start target HH-backscatter processing...'
     call backscatter_hh_retrieval(n, x, cx)
     write(*, '(a)') ' INFO::runua:'//&
          '...HH-backscatter processing DONE'

     write(*, '(a)') ' INFO::runua:'//&
          'start target FAPAR processing...'
     call fapar_retrieval(n, x, cx)
     write(*, '(a)') ' INFO::runua:'//&
          '...FAPAR processing DONE'
  endif

  !-- dispose memory
  deallocate( x, sx, cx )

3333 FORMAT (2x, a15,2x,a)
end program runua
