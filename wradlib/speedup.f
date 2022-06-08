! Copyright (c) wradlib developers.
! Distributed under the MIT License. See LICENSE.txt for more info.
C FILE: SPEEDUP.F
      SUBROUTINE F_UNFOLD_PHI(PHIDP, RHO, GRADPHI, STDARR, BEAMS, RS, W)
C
C     FORTRAN implementation of phase unfolding
C
      INTEGER beams, rs, w
      REAL*4 phidp(beams,rs)
      REAL*4 rho(beams,rs),gradphi(beams,rs), stdarr(beams,rs)
      INTEGER beam, j, count1, count2, k, l, w_
      REAL*4 ref
Cf2py intent(in) beams, rs, rho, gradphi, stdarr, w
Cf2py intent(in,out,overwrite) phidp

      w_ = w - 1

C   iterate over all beams
      DO beam=1, beams

         IF ( ALL(phidp(beam,1:(rs-w_))==0) ) THEN
            PRINT *, "empty beam at:", beam
            EXIT
         ENDIF

C        Determine location where meaningful phidpDP profile begins
         DO j=1, rs-w_
            count1 = 0
            count2 = 0
            DO i=j, j+w_
               IF (stdarr(beam,i)<5) THEN
                  count1 = count1 + 1
               ENDIF
               IF (rho(beam,i)>0.9) THEN
                  count2 = count2 + 1
               ENDIF
            ENDDO
C            print *, count1, count2
            IF ((count1==w) .AND. (count2==w)) THEN
               EXIT
            ENDIF
         ENDDO
         IF (j > (rs-w_)) THEN
            j = rs-w_
         ENDIF

C       Now start to check for phase folding
        ref = SUM( phidp(beam,j:j+w_) ) / w
        DO k=j+w_, rs
            sumup = SUM( stdarr(beam,k-w_:k) )
            slopeok = 1
            DO l=k-w_, k
               IF ((gradphi(beam,l)<-5) .OR. (gradphi(beam,l)>20)) THEN
                  slopeok = 0
               ENDIF
            ENDDO
            IF ((slopeok==1) .AND. (sumup<15.)) THEN
               ref = ref + gradphi(beam, k)*0.5
               IF (phidp(beam,k) - ref < -80.) THEN
                  IF (phidp(beam,k) < 0) THEN
                     phidp(beam,k) = phidp(beam,k) + 360.
                  ENDIF
               ENDIF
            ELSE IF (phidp(beam,k) - ref < -80.) THEN
                IF (phidp(beam,k) < 0) THEN
                   phidp(beam,k) = phidp(beam,k) + 360.
                ENDIF
            ENDIF
         ENDDO
      ENDDO
      END
