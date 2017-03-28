\version "2.12.0"
\header { 
  title = "Sinfonia K.97, III" 	
  composer = "Mozart" 
  piece = "Trio" 
}

\new ChoirStaff <<
  \new Staff {
    \set Staff.instrumentName = \markup {"Vl.I"}
    \relative c'{
      \time 3/4 
      \key g \major  
      \set Score.currentBarNumber = #25
      \bar ""

      \repeat volta 2 {
        b'-.\p a-. b-. 
        c2(\fp a4)
        b4-. a-. b-.
        c2(\fp a4)
        d-. c-. d-.
        e2(\fp c4)
        b d8 c b a
        g2 r4 
      } 
      \repeat volta 2 {
        g'-.\p fis-. e-.
        a(\f g fis )
        g-.\p fis-. e-.
        a(\f g fis)
        c?-.\p c-. c-.
        b ( d g )
        e d8 c b a
        g2 r4 
      }
    }
  }

  \new Staff {
    \set Staff.instrumentName = \markup {"Vl.II"}
    \relative c''{
      \time 3/4 
      \key g \major  
      \repeat volta 2 {
        g8\p d fis d g d
        a'\fp d, a' d, fis d
        g d fis d g d
        a'\fp d, a' d, fis d
        b' d, a' d, b' d,
        c'\fp d, c' d, a' d,
        g d b' a g fis
        g d c b a g 
      } 
      \repeat volta 2 {
        b'\p d a d g, d'
        fis,\f d' e, cis' d, d'
        b\p d a d g, d'
        fis,\f d' e, cis' d, d'
        fis,\p d' e, d' d, d'
        d, d' b d g, d'
        e, c' b a g fis
        g2 r4
      }
    }
  }

  \new Staff {
    \set Staff.instrumentName = \markup {"Vla."}
    \relative c'{
      \time 3/4 
      \key g \major  
      \clef alto
      \repeat volta 2 {
        b4-.\p a4-. b4-.
        c2( a4)
        b4-. a4-. b4-.
        c2(\fp a4)
        d4-. c4-. d4-.
        e2(\fp c4)
        b4 d8 c b a
        g4 r r 
      } 
      \repeat volta 2 {
        r2.
        a'4(\f g fis)
        r2.
        a4(\f g fis)
        c!4-.\p c4-. c4-.
        b4( e g)
        r4 d d
        d2 r4
      }
    }
  }

  \new Staff {
    \set Staff.instrumentName = \markup {\column{"Vc." "D.B."}}
    \relative c'{
      \time 3/4 
      \key g \major  
      \clef bass
      \repeat volta 2 {
        g4\p r r
        r d'\fp d,
        g4 r r
        r d'\fp d,
        g4 r r
        r fis fis
        g4 d' d,
        g4 r r 
      } 
      \repeat volta 2 {
        r2.
        d4(\f e fis)
        r2.
        d4(\f e fis)
        d4-.\p e-. fis-.
        g r b, c d d 
        g,2 r4
      }
    }
  }
>>