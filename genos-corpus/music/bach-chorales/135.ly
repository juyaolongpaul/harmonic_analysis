\version "2.10.33"

\header {
  title = "135 - Gott der Vater wohn uns bei"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    a4 a4 b4 cis4 
    d4 d4 cis4 b4 
    d4 a4 fis4 g4 
    fis4 e4 d2 }
  \repeat volta 2 {
    d4 e4 fis4. e8   %5
    d4 e4 fis4 d'4 
    cis4 d4 e4. d8 
    cis4 b4 a4 a4 
    a4 d4 cis4 a4 
    b8 a8 g4 fis4 e4   %10
    fis4 g4 a4. g8 
    fis4 e4 d2
  }
  d4 e4 fis4. e8 
  d4 e4 fis4 a4 
  a4 a4 fis4 g4 
  fis4 e4 d2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    fis4 fis4 g4 g4 
    a4 a8 gis8 a4 g4 
    fis4 e4 e8 d4 cis8 
    d4. cis8 a2 }
  \repeat volta 2 {
    a4 a8 e'4 d4 cis8   %5
    b4 b4 cis4 d8 e8 
    fis4 fis8 b4 a16 gis16 a4~ 
    a4 gis4 e4 e4 
    fis4 fis4 e4 d8 e8 
    fis4. e4 dis8 e4~   %10
    e8 d4 cis4 d4 e8~ 
    e8 d4 cis8 a2
  }
  a4 a8 e'4 d4 cis8 
  d4 b8 e8 d4 d4~ 
  d8 e8 fis8 e8 d4. e8~ 
  e8 d4 cis8 a2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    d4 d4 d4 e4 
    fis8 e8 d4 e4 e4 
    d4 e8 a,8 a4 g4 
    a4. a16 g16 fis2 }
  \repeat volta 2 {
    fis4 a4 a4 a8 g8   %5
    fis4 b4 ais4 b4~ 
    b8 a8 fis'4 e4 cis8 d8 
    e4. d8 cis4 cis4 
    d4 a4 a8 g8 fis4 
    b4 b4 b4 b8 a8   %10
    a4 g4 fis8 a8 b4 
    a4. g8 fis2
  }
  a4 a4 a4 a4 
  a8 fis8 g8 a8 a4 fis8 g8 
  a8 b8 cis4 d8 cis8 b4 
  a4. g8 fis2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    d4 d4 g8 fis8 e4 
    d8 cis8 b4 a4 e'4 
    b4 cis4 d4 e4 
    fis8 g8 a4 d,2 }
  \repeat volta 2 {
    d4 cis4 d4 a4   %5
    b8 a8 g4 fis4 b4 
    fis'4 d4 cis4 fis4 
    e4 e,4 a4 a'8 g8 
    fis8 e8 d4 a4 d4 
    dis4 e4 b4 cis4   %10
    d4 e4 fis4 g4 
    a4 a,4 d2
  }
  fis4 cis4 d4 a'8 g8 
  fis8 d4 cis8 d4 d8 e8 
  fis8 g8 a4 b8 a8 g4 
  a4 a,4 d2 
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}