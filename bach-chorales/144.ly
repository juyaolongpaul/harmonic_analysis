\version "2.10.33"

\header {
  title = "144 - Aus tiefer Not schrei ich zu dir 2"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c {
  \repeat volta 2 {
    r4 a''4 gis4 a4 
    b4 b4 a4 b4 
    cis2 r4 d4 
    cis4 b4 a4 gis8 a8 
    b2 a2 }  %5
  r4 a4 b8 cis8 d4 
  cis8 b8 a4 b8 a8 gis4 
  fis2 r4 gis4 
  e4 e4 b'4 cis4 
  d8 cis8 b4 a2   %10
  r4 e'4 cis4 cis4 
  b4 a4 gis4 a4 
  b2 a2 
  
}

alto = \relative c {
  \repeat volta 2 {
    r4 e'4 d4 e4 
    fis4 cis4 cis4 fis4 
    eis2 r4 fis4 
    e8 cis8 dis8 eis8 fis4 e4 
    fis4 e8 d8 cis2 }  %5
  r4 e4 fis4 eis8 fis8 
  gis4 cis,4 fis4. eis8 
  cis2 r4 dis4 
  e8 d4 cis8 dis8 e8 e4 
  d8 fis8 g2 fis4   %10
  r4 b4. a16 gis16 a4~ 
  a8 gis16 fis16 e4 e4 e4 
  e4. d8 cis2 
  
}

tenor = \relative c {
  \repeat volta 2 {
    r4 cis'4 b4 a4~ 
    a4 gis4 a4 gis8 fis8 
    gis2 r4 a4 
    a4 b4 cis4 b8 a8 
    a4 gis4 e2 }  %5
  r4 cis'4 d8 cis8 b8 a8 
  gis4 a8 gis8 fis4 cis'8 b8 
  a2 r4 gis4 
  gis4 a4 fis8 gis8 g8 fis8 
  fis8 d8 d'4 d2   %10
  r4 e4 e4 cis8 d16 e16 
  fis8 e8 a,4 d8 b8 a4~ 
  a4 gis4 e2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    r4 a4 b4 cis4 
    dis4 eis4 fis8 e8 d4 
    cis2 r4 fis8 gis8 
    a4. gis8 fis8 e8 d8 cis8 
    d8 b8 e4 a,2 }  %5
  r4 a'2 gis8 fis8 
  eis4 fis8 e8 d8 b8 cis4 
  fis,2 r4 bis4 
  cis8 b8 a4. gis8 ais4 
  b4. cis8 d2   %10
  r4 gis,4 a8 cis8 fis8 e8 
  d4. cis8 b8 e8 cis8 a8 
  e'4 e,4 a2 
  
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