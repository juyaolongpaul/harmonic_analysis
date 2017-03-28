\version "2.10.33"

\header {
  title = "310 - Mach's mit mir, Gott, nach deiner"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 e8  fis8 
    gis4 a4 b4 b4 
    a4 gis4 fis4 b4 
    cis4 dis4 e4 dis8 cis8 
    cis2 b4 } b4 
  e4 dis8 cis8 cis4 b4   %5
  a4 gis8 fis8 fis4 b4 
  a4 gis4 fis8 gis8 a4 
  gis4 fis4 e4 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    e4 e4 e4 e4 
    e8 dis8 e4 dis4 fis4 
    fis4 ais4 b8 fis8 fis8 gis8 
    gis4 ais4 fis4 } gis4 
  cis4 b4. a4 gis8~   %5
  gis8 fis8 e4 dis4 d4 
  c4 b4 a8 b8 c4 
  b8 e4 dis8 b4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 gis8  a8 
    b4 cis4 d4 cis8 b8 
    cis4 b4 b4 dis4 
    cis4 fis4 b,8 cis8 dis8 e8 
    e8 dis8 cis4 dis4 } e4 
  gis4 gis4 gis8 fis8 fis8 dis8   %5
  e8 b8 b4 b4 gis8 fis8 
  e4 eis4 fis4 fis4 
  gis8. a16 b8 fis8 gis4 
  
}

baixo = \relative c, {
  \repeat volta 2 {
    \partial 4 e4 
    e'4 cis4 gis4 cis4 
    fis4 gis8 a b4 b 
    ais4 gis8 fis8 gis8 ais8 b4 
    e,4 fis4 b,4 } e8 dis8 
  cis4 gis4 a4 b4   %5
  cis8 dis8 e4 b4 b4 
  c4 cis4 d4 dis4 
  e4 b4 e4 
  
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