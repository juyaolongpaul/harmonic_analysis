\version "2.10.33"

\header {
  title = "73 - Herr Jesu Christ, du höchstes Gut"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 fis4 g4 a4 
    bes4 a4 g4 g4 
    a4 a4 d4 c4 
    bes2 a4 } d4 
  c4 bes4 a8 bes8 c4   %5
  bes4 a4 g4 bes4 
  a4 g4 f8 e8 d8 e8 
  f4 g4 a4 a4 
  bes4 c4 d4. c8 
  bes4 c8 bes8 a2   %10
  g2.
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    ees4 d8 c8 bes4 ees4 
    d4 d4 d4 d4 
    ees4 d4 d4 g8 fis8 
    g2 fis4 } g4 
  g8 f8 f8 ees8 ees8 d8 c8 a'8   %5
  fis8 g4 fis8 d4 g8 f8 
  e8 d8 d8 cis8 d8 e8 f8 cis8 
  d8 f4 e8 f4 f4 
  f8 g8 a4 a4 d,4~ 
  d4 c8 d8 ees4 d8 c8   %10
  b2.
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    a4 a4 g4 c4 
    fis,8 g4 fis8 bes4 bes4 
    c4 c4 b4 c4 
    d4 e8 g,8 d'4 } d4 
  ees8 f8 g8 bes,8 c8 d8 g,8 c8   %5
  d8 ees8 a,8 d8 bes4 d4 
  cis8 d8 e4 a,4. g8 
  a8 c8 d8 c8 c4 c4 
  d4 ees4 f4 fis4 
  g4 g,2 fis4   %10
  d2.
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    c,4 d4 ees4 d8 c8 
    d4 d4 g,4 g'4 
    g4 fis4 f4 ees4 
    d4 cis4 d4 } bes'4~ 
  bes8 a8 g4. f8 ees4   %5
  d8 c8 d4 g,4 g'4~ 
  g8 f8 e4 d2~ 
  d8 a8 bes8 c8 f,4 f'8 ees8 
  d8 ees8 d8 c8 bes8 c8 bes8 a8 
  g8 bes8 ees8 d8 c8 a8 d4   %10
  g,2. 
  
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