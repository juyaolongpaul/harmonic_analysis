\version "2.10.33"

\header {
  title = "126 - Durch Adams Fall ist ganz verderbt"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    a4 a4 g4 a4 
    f4 e4 d4 a'8 b8 
    c4 d4 a4 b4 
    c4 b4 a4 } c4 
  c4 c8 bes8 a8 g8 f8 g8   %5
  a4 g4 f4 a4 
  a4 a4 g4 f8 e8 
  e2 d4 a'4 
  a4 g4 a4 b4 
  c8 b8 a4 g4 d'4   %10
  e4 d4 a8 b8 c4 
  b4 a4 b2 
  a1 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    f4 f4 f4 e4~ 
    e8 d4 cis8 d4 f4 
    e4 d8 e8 f4 f4 
    e4 d4 cis4 } a'4 
  g8 f8 e8 d8 c4 d8 e8   %5
  f4 e4 f4 c8 d8 
  e4 d4 d4 d4 
  d4 cis4 d4 e4 
  f4. e8 f8 a8 g8 f8 
  e8 d8 c4 b4 g'4   %10
  g4 f8 e8 d4 e4 
  e8 d8 c4 f4 e8 d8 
  cis1 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    d4 d4 d4 a4 
    a4. g8 f4 d'4 
    c8 b8 a4 d4 d4 
    gis,8 a4 gis8 a4 } e'4 
  c4 g4 a4 a8 bes8   %5
  c4. bes8 a4 a4 
  a8 g8 f4 g8 a8 bes4 
  a4. g8 f4 a4 
  d8 c8 bes4 c4 d4 
  g,4. fis8 g4 b4   %10
  c8 b8 a8 gis8 a4 a4 
  gis4 a2 gis4 
  a1 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 cis4 
    d8 e8 f4 b,4 cis4 
    d4 a4 d4 d4 
    a'8 g8 f8 e8 d8 c8 b8 a8 
    e'2 a,4 } a'4 
  e8 d8 c4 f8 e8 d4   %5
  a8 bes8 c4 f,4 f'4 
  cis4 d8 c8 bes8 a8 g4 
  a2 d4 cis4 
  d4 g4 f8 e8 d4 
  c4 d4 g,4 g'4   %10
  c,4 d8 e8 f4 c8 d8 
  e4 f8 e8 d4 e4 
  a,1 
  
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