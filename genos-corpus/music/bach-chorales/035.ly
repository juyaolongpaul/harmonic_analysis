\version "2.10.33"

\header {
  title = "35 - Gott des Himmels und der Erden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    a4 b4 cis8 d8 e4 
    a,4 gis4 fis4 e4 
    fis4 gis4 a4 b4 
    cis8 d8 b4 a2 }
  cis4 cis4 b4 cis4   %5
  d4 cis8 b8 b8 a8 b4 
  fis4 gis4 a4 b4 
  cis8 d8 b4 a8 gis8 a4 
  
}

alto = \relative c' {
  \repeat volta 2 {
    e4 e4 a4. gis8~ 
    gis8 fis4 e4 dis8 b4 
    d8 cis8 b4 e4 e4 
    e8 fis8 e4 e2 }
  a4 a4 e8 fis8 gis4   %5
  fis4 e8 fis8 gis8 fis8 gis4 
  cis,8 d8 e4. d16 cis16 fis4 
  e4 e4 e8 d8 e4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    cis4 b4 a8 b8 cis8 d8 
    e8 b8 b4 b8. a16 gis4 
    b8 cis8 d8 e8 a,4 e4 
    a4. gis8 cis2 }
  e4. dis8 e8 d8 cis8 b8   %5
  a8 b8 cis8 dis8 e4 e4 
  ais,8 b b4 a4 a4~ 
  a8 b8 gis8 d'8 cis8 b8 cis4 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    a4 gis4 fis4 cis8 b8 
    cis8 dis8 e4 b4 e4 
    b4 b'8 cis16 d16 cis8 b8 a8 gis8 
    a8 d,8 e4 a,2 }
  a'8 gis8 fis4 gis8 fis8 e4   %5
  fis8 gis8 a4 e4 e4 
  e8 d8 cis8 b8 cis8 fis8 e8 d8 
  e4 e,4 a4 a4 
  
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