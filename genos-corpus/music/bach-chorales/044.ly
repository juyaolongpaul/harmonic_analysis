\version "2.10.33"

\header {
  title = "44 - Mach's mit mir, Gott, nach deiner Güt'"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  e8 
    fis4 g4 a4 a4 
    g4 fis4 e4\fermata a4 
    b4 cis4 d4 cis4 
    b2 a4\fermata } a4 
  d4 cis4 b4 a4   %5
  g4 fis4 e4\fermata a4 
  g4 fis4 e8 fis8 g4 
  fis4 e4 d\fermata
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    d4 d4 e4 d8 cis8 
    b8 cis8 d4 cis4 e4 
    fis8 gis8 a4 b4 a4~ 
    a4 gis4 e4 } e4 
  fis4 e4 d8 e8 fis4   %5
  e4 a,8 b8 cis4 fis8 e8 
  d8 e4 d4 cis8 d4 
  d4 cis4 a 
  
}

tenor = \relative c {
  \repeat volta 2 {
    \partial 4 fis8  g8 
    a4 g8 fis8 e4 fis4 
    g4 a4 a4 a4 
    d4 e4 e4 e4 
    fis4 e8 d8 cis4 } cis8 b8 
  a8 fis8 g8 a8 b4 b4   %5
  b8 cis8 d4 a4 cis4 
  b8 a8 a4 a4 g4 
  a4. g8 fis4
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 d4 
    d8 cis8 b4 cis4 d4 
    e4 fis8 g8 a4\fermata cis4 
    b4 a4 gis4 a4 
    d,4 e4 a,4\fermata } a'8 g8 
  fis8 d8 e8 fis8 g8 fis8 e8 dis8   %5
  e4 fis8 gis8 a4\fermata fis4 
  b,8 cis8 d4 a4 b4 
  a8 g8 a4 d,\fermata 
  
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