\version "2.10.33"

\header {
  title = "191 - Von Gott will ich nicht lassen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    a4 b4 c4 d4 
    b2 g4 g4 
    a4 a4 b4 b4 
    e,2. } e'4 
  d4 c4 c4 b4   %5
  c2. b4 
  c4 d4 e4 e4 
  d2 b4 b4 
  c4 b4 a4 a4 
  b4 a4 g4 f4   %10
  e4 a4 a4 gis4 
  a2.
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    e4 e4 e4 d4 
    d8 e8 fis4 e4 e4 
    c4 d4 d8 f8 e8 d8 
    c2. } a'4 
  a4 a4 g4 g4   %5
  g2. g4 
  g4 f4 g4 a8 g8 
  fis8 e8 fis4 d4 g4 
  g4 e4 c4 f4~ 
  f8 e8 fis4 e4 d4   %10
  c4 d4 e4 e4 
  e2.
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    c4 b4 a4 a4 
    g4 fis4 b4 c4 
    a4 f4 e4 e4 
    a2. } c4 
  d4 f4 e4 d4   %5
  e2. d4 
  c4. b8~ b g8 c4
  c4 b8 a8 g4 d'4 
  c4 g4 a8 b8 c4 
  d4 c4 b4 g4   %10
  g4 a8 b8 c8 b16 a16 b8 b8 
  cis2.
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 a4 
    a'4 gis4 a4 fis4 
    g4 dis4 e4 c4 
    f4 d4 gis,4 gis4 
    a2. } a'4 
  f4 d4 g4 g,4   %5
  c2. g'4 
  e4 d4 c4 a4 
  d2 g,4 g'4 
  e4 c4 f4 e4 
  d4 dis4 e4 b4   %10
  c4 f4 e4 e,4 
  a2.
  
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